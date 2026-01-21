use anyhow::{anyhow, Result};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use std::sync::Arc;

use crate::collection_manager::sides::read::{ReadSide, SearchAnalyticEventOrigin, SearchRequest};
use crate::types::{CollectionId, NLPSearchRequest, ReadApiKey, SearchParams};

#[pyclass]
pub struct SearchService {
    read_side: Arc<ReadSide>,
    api_key: ReadApiKey,
    collection_id: CollectionId,
}

#[pymethods]
impl SearchService {
    fn search(&self, py: Python, params: Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        let search_params: SearchParams =
            serde_pyobject::from_pyobject(params.clone()).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to deserialize search params: {e}"
                ))
            })?;

        let read_side = self.read_side.clone();
        let api_key = self.api_key.clone();
        let collection_id = self.collection_id;

        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {
                read_side
                    .search(
                        &api_key,
                        collection_id,
                        SearchRequest {
                            search_params,
                            analytics_metadata: None,
                            interaction_id: None,
                            search_analytics_event_origin: Some(SearchAnalyticEventOrigin::MCP),
                        },
                    )
                    .await
            })
        });

        match result {
            Ok(search_result) => {
                let json = serde_json::to_value(&search_result).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Failed to serialize MCP search result: {e}"
                    ))
                })?;

                serde_pyobject::to_pyobject(py, &json)
                    .map(|obj| obj.unbind())
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Failed to convert to Python: {e}"
                        ))
                    })
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "MCP Search failed: {e}"
            ))),
        }
    }

    fn search_json(&self, params_json: String) -> PyResult<String> {
        let search_params: SearchParams = serde_json::from_str(&params_json).map_err(|e| {
            tracing::error!("Failed to parse MCP search params JSON: {}", e);
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to parse MCP search params JSON: {e}"
            ))
        })?;

        let read_side = self.read_side.clone();
        let api_key = self.api_key.clone();
        let collection_id = self.collection_id;

        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {
                read_side
                    .search(
                        &api_key,
                        collection_id,
                        SearchRequest {
                            search_params,
                            analytics_metadata: None,
                            interaction_id: None,
                            search_analytics_event_origin: Some(SearchAnalyticEventOrigin::MCP),
                        },
                    )
                    .await
            })
        });

        match result {
            Ok(search_result) => serde_json::to_string(&search_result).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to serialize search result: {e}"
                ))
            }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "MCP Search failed: {e}"
            ))),
        }
    }

    fn nlp_search(&self, py: Python, params: Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        let nlp_request: NLPSearchRequest =
            serde_pyobject::from_pyobject(params.clone()).map_err(|e| {
                tracing::error!("Failed to deserialize MCP NLP search params: {}", e);
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to deserialize MCP NLP search params: {e}"
                ))
            })?;

        let read_side = self.read_side.clone();
        let api_key = self.api_key.clone();
        let collection_id = self.collection_id;

        // Release the GIL before blocking on async operations
        // @todo: remove this once on Python v3.14
        #[allow(deprecated)]
        let result = py.allow_threads(|| {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async move {
                    let logs = read_side.get_hook_logs();
                    let log_sender = logs.get_sender(&collection_id);
                    let result = read_side
                        .nlp_search(
                            axum::extract::State(read_side.clone()),
                            api_key,
                            collection_id,
                            nlp_request,
                            log_sender,
                        )
                        .await;

                    result
                })
            })
        });

        match result {
            Ok(nlp_result) => {
                let json = serde_json::to_value(&nlp_result).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Failed to serialize MCP NLP result: {e}"
                    ))
                })?;

                serde_pyobject::to_pyobject(py, &json)
                    .map(|obj| obj.unbind())
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Failed to convert to Python: {e}"
                        ))
                    })
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "MCP NLP search failed: {e}"
            ))),
        }
    }
}

pub struct McpService {
    instance: Py<PyAny>,
}

impl McpService {
    pub fn new(
        read_side: Arc<ReadSide>,
        collection_id: CollectionId,
        api_key: ReadApiKey,
        collection_description: String,
    ) -> PyResult<Self> {
        Python::attach(|py| {
            let search_service = SearchService {
                read_side,
                api_key,
                collection_id,
            };
            let search_service_py = Py::new(py, search_service)?;
            let mcp_module = py.import("src.mcp.mcp")?;
            let mcp_class = mcp_module.getattr("MCP")?;
            let instance = mcp_class.call1((search_service_py, collection_description))?;

            Ok(McpService {
                instance: instance.into(),
            })
        })
    }

    pub fn list_tools(&self) -> Result<Value> {
        Python::attach(|py| {
            let instance = self.instance.bind(py);
            let tools = instance.call_method0("list_tools")?;
            let tools_json: Value = serde_pyobject::from_pyobject(tools)
                .map_err(|e| anyhow!("Failed to convert tools list: {e}"))?;

            Ok(tools_json)
        })
    }

    pub fn call_tool(&self, tool_name: &str, arguments: Value) -> Result<Value> {
        Python::attach(|py| {
            let instance = self.instance.bind(py);
            let args_py = serde_pyobject::to_pyobject(py, &arguments)
                .map_err(|e| anyhow!("Failed to convert arguments to Python: {e}"))?;

            let result = instance.call_method1("call_tool", (tool_name, args_py))?;
            let result_json: Value = serde_pyobject::from_pyobject(result)
                .map_err(|e| anyhow!("Failed to convert result from Python: {e}"))?;

            Ok(result_json)
        })
    }

    pub fn handle_jsonrpc(&self, request_str: String) -> Result<String> {
        Python::attach(|py| {
            let instance = self.instance.bind(py);
            let result = instance.call_method1("handle_jsonrpc_request", (request_str,))?;
            let response_str: String = result
                .extract()
                .map_err(|e| anyhow!("Failed to extract JSON-RPC response string: {e}"))?;
            Ok(response_str)
        })
    }
}
