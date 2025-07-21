use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq)]
pub struct ContextComponent {
    pub source_ids: Vec<String>,
    pub threshold: f32,
    pub max_documents: usize,
    pub fill_remaining: bool,
    pub is_exclusion: bool,
}

#[derive(Debug, Clone)]
pub struct ParseResult {
    pub components: Vec<ContextComponent>,
    pub success: bool,
    pub error_message: Option<String>,
    pub error_position: Option<usize>,
}

#[derive(Debug)]
pub enum ParseError {
    InvalidSyntax(String),
    InvalidThreshold(String),
    InvalidMaxDocuments(String),
    EmptySourceList,
    MissingAtSymbol,
    MissingColon,
    UnexpectedCharacter(char, usize),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::InvalidSyntax(msg) => write!(f, "Invalid syntax: {msg}"),
            ParseError::InvalidThreshold(val) => write!(f, "Invalid threshold value: {val}"),
            ParseError::InvalidMaxDocuments(val) => write!(f, "Invalid max documents: {val}"),
            ParseError::EmptySourceList => write!(f, "Source list cannot be empty"),
            ParseError::MissingAtSymbol => write!(f, "Missing @ symbol for threshold"),
            ParseError::MissingColon => write!(f, "Missing : symbol for max documents"),
            ParseError::UnexpectedCharacter(ch, pos) => {
                write!(f, "Unexpected character '{ch}' at position {pos}")
            }
        }
    }
}

pub struct RAGAtParser;

impl RAGAtParser {
    pub fn parse(notation: &str) -> ParseResult {
        match Self::parse_internal(notation) {
            Ok(components) => ParseResult {
                components,
                success: true,
                error_message: None,
                error_position: None,
            },
            Err(error) => ParseResult {
                components: Vec::new(),
                success: false,
                error_message: Some(error.to_string()),
                error_position: None, // Could be enhanced to track positions
            },
        }
    }

    fn parse_internal(notation: &str) -> Result<Vec<ContextComponent>, ParseError> {
        let notation = notation.trim();
        if notation.is_empty() {
            return Ok(Vec::new());
        }

        let component_strings: Vec<&str> = notation.split(';').collect();
        let mut components = Vec::new();

        for component_str in component_strings {
            let component = Self::parse_component(component_str.trim())?;
            components.push(component);
        }

        Ok(components)
    }

    fn parse_component(component_str: &str) -> Result<ContextComponent, ParseError> {
        if component_str.is_empty() {
            return Err(ParseError::InvalidSyntax("Empty component".to_string()));
        }

        let (is_exclusion, remaining) = if let Some(remaining) = component_str.strip_prefix('!') {
            (true, remaining)
        } else {
            (false, component_str)
        };

        let at_pos = remaining.find('@').ok_or(ParseError::MissingAtSymbol)?;

        let source_part = &remaining[..at_pos].trim();
        let params_part = &remaining[at_pos + 1..];

        let source_ids = Self::parse_source_ids(source_part)?;

        let (threshold, max_documents, fill_remaining) = Self::parse_params(params_part)?;

        Ok(ContextComponent {
            source_ids,
            threshold,
            max_documents,
            fill_remaining,
            is_exclusion,
        })
    }

    fn parse_source_ids(source_part: &str) -> Result<Vec<String>, ParseError> {
        let source_part = source_part.trim();

        if source_part.is_empty() {
            return Err(ParseError::EmptySourceList);
        }

        let inner = source_part;
        if inner.is_empty() {
            return Err(ParseError::EmptySourceList);
        }

        let ids: Vec<String> = inner
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if ids.is_empty() {
            return Err(ParseError::EmptySourceList);
        }

        Ok(ids)
    }

    fn parse_params(params_part: &str) -> Result<(f32, usize, bool), ParseError> {
        let colon_pos = params_part.find(':').ok_or(ParseError::MissingColon)?;

        let threshold_str = &params_part[..colon_pos];
        let max_docs_str = &params_part[colon_pos + 1..];

        let threshold = threshold_str
            .trim()
            .parse::<f32>()
            .map_err(|_| ParseError::InvalidThreshold(threshold_str.to_string()))?;

        let (max_docs_str, fill_remaining) = if let Some(remaining) = max_docs_str.strip_suffix('+')
        {
            (remaining, true)
        } else {
            (max_docs_str, false)
        };

        let max_documents = max_docs_str
            .trim()
            .parse::<usize>()
            .map_err(|_| ParseError::InvalidMaxDocuments(max_docs_str.to_string()))?;

        Ok((threshold, max_documents, fill_remaining))
    }

    pub fn validate_sources(
        components: &[ContextComponent],
        available_indexes: &HashSet<String>,
    ) -> Result<(), String> {
        for (i, component) in components.iter().enumerate() {
            for source_id in &component.source_ids {
                if !available_indexes.contains(source_id) {
                    return Err(format!(
                        "Component {i}: Source ID '{source_id}' not found in available indexes"
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn extract_all_source_ids(components: &[ContextComponent]) -> HashSet<String> {
        components
            .iter()
            .flat_map(|c| &c.source_ids)
            .cloned()
            .collect()
    }
}

#[derive(Debug)]
pub enum GeneralRagAtError {
    ParseError(String),
    InvalidIndexId(String),
    ReadError,
}

impl ContextComponent {
    pub fn new(source_ids: Vec<String>, threshold: f32, max_documents: usize) -> Self {
        Self {
            source_ids,
            threshold,
            max_documents,
            fill_remaining: false,
            is_exclusion: false,
        }
    }
}
