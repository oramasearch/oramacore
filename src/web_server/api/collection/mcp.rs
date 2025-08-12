use rmcp::{
    handler::server::tool::{Parameters, ToolRouter},
    tool_handler, tool_router,
};

#[derive(Clone)]
pub struct StructuredOutputServer {
    tool_router: ToolRouter<Self>,
}

#[tool_handler(router = self.tool_router)]
impl rmcp::ServerHandler for StructuredOutputServer {}

impl Default for StructuredOutputServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_router(router = tool_router)]
impl StructuredOutputServer {
    pub fn new() -> Self {
        Self {
            tool_router: ToolRouter::new(),
        }
    }

    #[tool(
        name = "search",
        description = "Perform a search operation on all the indexes"
    )]
    pub async fn search(&self, params: Parameters) {}
}
