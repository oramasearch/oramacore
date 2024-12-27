use crate::long_term_memory::knowledge_graph::Metadata;

struct ConversationAnalyzer {
    patterns: Vec<Box<dyn Fn(&str) -> Option<Metadata>>>,
}

impl ConversationAnalyzer {
    fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    fn add_pattern<F>(&mut self, pattern: F)
    where
        F: Fn(&str) -> Option<Metadata> + 'static,
    {
        self.patterns.push(Box::new(pattern));
    }

    fn analyze(&self, message: &str) -> Vec<Metadata> {
        self.patterns
            .iter()
            .filter_map(|pattern| pattern(message))
            .collect()
    }
}
