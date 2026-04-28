use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tiktoken_rs::o200k_base;

use super::claude::{
    ContentBlock, CreateMessageParams as ClaudeCreateMessageParams, Message, MessageContent, Role,
    Thinking, Tool, ToolChoice, default_max_tokens,
};

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    Low = 256,
    #[default]
    Medium = 256 * 8,
    High = 256 * 8 * 8,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OaiToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OaiToolCall {
    pub id: String,
    #[serde(rename = "type", default)]
    pub type_: String,
    pub function: OaiToolCallFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OaiMessage {
    pub role: Role,
    pub content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OaiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl OaiMessage {
    fn to_claude_message(self) -> Message {
        match self.role {
            Role::Tool => {
                let text = extract_text_content(&self.content);
                let tool_result = ContentBlock::ToolResult {
                    tool_use_id: self.tool_call_id.unwrap_or_default(),
                    content: Value::String(text),
                    cache_control: None,
                    is_error: None,
                };
                Message {
                    role: Role::User,
                    content: MessageContent::Blocks {
                        content: vec![tool_result],
                    },
                }
            }
            Role::Assistant => {
                if let Some(tool_calls) = self.tool_calls {
                    let mut blocks: Vec<ContentBlock> = Vec::new();
                    if let Some(text) = extract_text_content_opt(&self.content) {
                        if !text.is_empty() {
                            blocks.push(ContentBlock::text(text));
                        }
                    }
                    for tc in tool_calls {
                        let input: Value =
                            serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Null);
                        blocks.push(ContentBlock::ToolUse {
                            id: tc.id,
                            name: tc.function.name,
                            input,
                            cache_control: None,
                            caller: None,
                        });
                    }
                    Message {
                        role: Role::Assistant,
                        content: MessageContent::Blocks { content: blocks },
                    }
                } else {
                    Message::new_text(Role::Assistant, extract_text_content(&self.content))
                }
            }
            Role::System => {
                Message::new_text(Role::System, extract_text_content(&self.content))
            }
            _ => {
                let text = extract_text_content(&self.content);
                Message::new_text(Role::User, text)
            }
        }
    }
}

fn extract_text_content(content: &Option<Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| {
                if v.get("type").and_then(|t| t.as_str()) == Some("text") {
                    v.get("text").and_then(|t| t.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn extract_text_content_opt(content: &Option<Value>) -> Option<String> {
    match content {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(Value::Array(arr)) => {
            let text: String = arr
                .iter()
                .filter_map(|v| {
                    if v.get("type").and_then(|t| t.as_str()) == Some("text") {
                        v.get("text").and_then(|t| t.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("");
            if text.is_empty() { None } else { Some(text) }
        }
        _ => None,
    }
}

impl From<CreateMessageParams> for ClaudeCreateMessageParams {
    fn from(params: CreateMessageParams) -> Self {
        let claude_messages: Vec<Message> = params
            .messages
            .into_iter()
            .map(|m| m.to_claude_message())
            .collect();
        let (systems, messages): (Vec<Message>, Vec<Message>) =
            claude_messages.into_iter().partition(|m| m.role == Role::System);
        let systems = systems
            .into_iter()
            .map(|m| m.content)
            .flat_map(|c| match c {
                MessageContent::Text { content } => vec![ContentBlock::text(content)],
                MessageContent::Blocks { content } => content,
            })
            .filter(|b| matches!(b, ContentBlock::Text { .. }))
            .map(|b| json!(b))
            .collect::<Vec<_>>();
        let system = (!systems.is_empty()).then(|| json!(systems));
        Self {
            max_tokens: (params.max_tokens.or(params.max_completion_tokens))
                .unwrap_or_else(default_max_tokens),
            system,
            messages,
            model: params.model,
            container: None,
            context_management: None,
            mcp_servers: None,
            stop_sequences: params.stop,
            thinking: params
                .thinking
                .or_else(|| params.reasoning_effort.map(|e| Thinking::new(e as u64))),
            temperature: params.temperature,
            stream: params.stream,
            top_k: params.top_k,
            top_p: params.top_p,
            tools: params.tools,
            tool_choice: params.tool_choice,
            metadata: params.metadata,
            output_config: None,
            output_format: None,
            service_tier: None,
            n: params.n,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct CreateMessageParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    pub messages: Vec<OaiMessage>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<Effort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
}

use super::claude::Metadata;

impl CreateMessageParams {
    pub fn count_tokens(&self) -> u32 {
        let bpe = o200k_base().expect("Failed to get encoding");
        let messages = self
            .messages
            .iter()
            .map(|msg| match &msg.content {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(arr)) => arr
                    .iter()
                    .filter_map(|v| v.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join(""),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join("\n");
        bpe.encode_with_special_tokens(&messages).len() as u32
    }
}
