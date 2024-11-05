use std::{
    collections::HashMap,
    fs::{self, DirEntry},
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

use collection_manager::{
    dto::{CreateCollectionOptionDTO, Limit, SearchParams, TypedField},
    CollectionManager, CollectionsConfiguration,
};
use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, LinkType, Tag, TagEnd};
use rocksdb::OptimisticTransactionDB;
use serde::Serialize;
use serde_json::json;
use storage::Storage;

use types::CodeLanguage;

fn main() -> anyhow::Result<()> {
    let storage_dir = "./tanstack";

    let _ = fs::remove_dir_all(storage_dir);
    let db = OptimisticTransactionDB::open_default(storage_dir).unwrap();
    let storage = Arc::new(Storage::new(db));

    let manager = CollectionManager::new(CollectionsConfiguration { storage });

    let collection_id = manager
        .create_collection(CreateCollectionOptionDTO {
            id: "tanstack".to_string(),
            description: None,
            language: None,
            typed_fields: vec![("code".to_string(), TypedField::Code(CodeLanguage::TSX))]
                .into_iter()
                .collect(),
        })
        .expect("unable to create collection");

    let all_files = get_md_files(
        "/Users/allevo/repos/rustorama/tanstack_example/tanstack_table/docs"
            .parse()
            .unwrap(),
    );

    let mut orama_documents = vec![];
    for file in all_files {
        let page = parse_from_file(file.path().to_path_buf());

        let page_header = page.heading.clone();
        for section in page.sections {
            for (language, code) in section.codes {
                if language != "tsx" {
                    continue;
                }

                let url = path_clean::clean(file.path().to_str().unwrap());
                let id = if let Some(heading) = &section.heading {
                    format!("{url:?}#{}", heading.text)
                } else {
                    url.to_string_lossy().to_string()
                };
                orama_documents.push(json!({
                    "id": id,
                    "url": url,
                    "language": "tsx",
                    "code": code,
                    "section": section.heading.clone(),
                    "page": page_header.clone(),
                }));
            }
        }
    }

    println!("orama_documents: {:#?}", orama_documents.len());

    manager.get(collection_id.clone(), |collection| {
        collection.insert_batch(orama_documents.try_into().unwrap())
    });

    let term = "SelectionTableState";

    println!(
        r###"

--- SEARCH EXAMPLE ---
term: {term}

"###
    );

    let output = manager.get(collection_id, |collection| {
        collection.search(SearchParams {
            term: term.to_string(),
            limit: Limit(10),
            boost: Default::default(),
            properties: Some(vec!["code".to_string()]),
        })
    });

    println!("{:#?}", output);

    Ok(())
}

fn get_md_files(dir: PathBuf) -> Vec<DirEntry> {
    let mut files = vec![];
    fn visit_dirs(dir: &Path, files: &mut Vec<DirEntry>) -> io::Result<()> {
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    visit_dirs(&path, files)?;
                } else if let Some(extension) = path.extension() {
                    if extension == "md" {
                        files.push(entry);
                    }
                }
            }
        }
        Ok(())
    }
    visit_dirs(&dir, &mut files).unwrap();

    files
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
enum Level {
    H1,
    H2,
    H3,
    H4,
    H5,
    H6,
}
impl From<HeadingLevel> for Level {
    fn from(level: HeadingLevel) -> Self {
        match level {
            HeadingLevel::H1 => Level::H1,
            HeadingLevel::H2 => Level::H2,
            HeadingLevel::H3 => Level::H3,
            HeadingLevel::H4 => Level::H4,
            HeadingLevel::H5 => Level::H5,
            HeadingLevel::H6 => Level::H6,
        }
    }
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
struct PageHeading {
    id: String,
    title: String,
    metadata: HashMap<String, String>,
}
#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
struct SectionHeading {
    level: Level,
    text: String,
}

#[derive(Debug, Serialize, Clone)]
struct Section {
    heading: Option<SectionHeading>,
    links: Vec<Link>,
    content: Vec<String>,
    codes: Vec<(String, String)>,
}
#[derive(Debug, Serialize, Clone)]
struct Link {
    absolute_url: PathBuf,
    title: Option<String>,
    id: Option<String>,
    text: String,
}
#[derive(Debug, Serialize, Clone)]
struct Page {
    heading: Option<PageHeading>,
    sections: Vec<Section>,
}

fn parse_from_file(file_path: PathBuf) -> Page {
    let file_content = std::fs::read_to_string(file_path.clone()).unwrap();
    parse(&file_content, file_path)
}

fn parse(file_content: &str, file_path: PathBuf) -> Page {
    let base_url = file_path.parent().unwrap().to_path_buf();

    let parser = pulldown_cmark::Parser::new(file_content);

    let mut page_heading = None;
    let mut sections = vec![];

    let mut current_section = Some(Section {
        heading: None,
        links: vec![],
        content: vec![],
        codes: vec![],
    });
    let mut current_heading = None;
    let mut current_link = None;
    let mut current_paragraph = None;
    let mut current_code_block = None;

    let mut first_header = true;

    for ev in parser {
        // println!("{:?}", ev);
        match ev.clone() {
            Event::Start(Tag::Heading { .. }) => {
                if !first_header && current_section.is_some() {
                    let section = current_section.take().unwrap();
                    sections.push(section);
                }
                assert!(current_paragraph.is_none());
                assert!(current_link.is_none());
                assert!(current_heading.is_none());
                current_heading = Some(vec![ev]);
            }
            Event::Start(Tag::Paragraph) => {
                assert!(current_paragraph.is_none());
                current_paragraph = Some(vec![]);
            }
            Event::Start(Tag::Link { .. }) => {
                assert!(current_link.is_none());
                current_link = Some(vec![ev]);
            }
            Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(_))) => {
                current_code_block = Some(vec![ev]);
            }
            Event::Start(Tag::CodeBlock(CodeBlockKind::Indented)) => {
                panic!("unexpected code block Indented");
            }
            Event::Text(_) => {
                if let Some(current_link) = current_link.as_mut() {
                    // assert!(current_heading.is_none(), "unexpected text in link {:?} at {file_content}", ev);
                    current_link.push(ev);
                } else if let Some(current_heading) = current_heading.as_mut() {
                    assert!(current_link.is_none());
                    assert!(current_paragraph.is_none());
                    current_heading.push(ev);
                } else if let Some(current_paragraph) = current_paragraph.as_mut() {
                    assert!(current_heading.is_none());
                    assert!(current_link.is_none());
                    current_paragraph.push(ev);
                } else if let Some(current_code_block) = current_code_block.as_mut() {
                    current_code_block.push(ev);
                } else {
                    // panic!("unexpected text {:?}", ev);
                }
            }
            Event::Code(_) => {
                if let Some(current_heading) = current_heading.as_mut() {
                    current_heading.push(ev);
                } else if let Some(current_paragraph) = current_paragraph.as_mut() {
                    current_paragraph.push(ev);
                } else {
                    // println!("{current_section:?}");
                    // println!("{current_heading:?}");
                    // println!("{current_link:?}");
                    // println!("{current_paragraph:?}");
                    // println!("{current_code_block:?}");
                    // panic!("unexpected code block {:?} {:?}", file_path, c);
                }
            }
            Event::End(TagEnd::CodeBlock) => {
                let mut current_code_block = current_code_block.take().unwrap();
                let lang = match current_code_block.remove(0) {
                    Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(k))) => k.to_string(),
                    _ => unreachable!(),
                };

                let code = match current_code_block.remove(0) {
                    Event::Text(ref text) => text.to_string(),
                    _ => unreachable!(),
                };

                assert_eq!(current_code_block.len(), 0);

                current_section.as_mut().unwrap().codes.push((lang, code));
            }
            Event::End(TagEnd::Link) => {
                let mut link = current_link.take().unwrap();

                let (absolute_url, title, id) = match link.remove(0) {
                    Event::Start(Tag::Link {
                        dest_url,
                        title,
                        id,
                        link_type,
                    }) => {
                        assert_eq!(link_type, LinkType::Inline);
                        let absolute_url = base_url.join(dest_url.to_string());
                        (absolute_url, title.to_string(), id.to_string())
                    }
                    _ => unreachable!(),
                };

                let mut texts = vec![];
                for l in link {
                    match l {
                        Event::Text(ref text) => texts.push(text.to_string()),
                        _ => unimplemented!("link child should be text"),
                    }
                }
                let text = texts.join("");

                if let Some(current_paragraph) = current_paragraph.as_mut() {
                    current_paragraph.push(Event::Text(pulldown_cmark::CowStr::Boxed(
                        text.clone().into_boxed_str(),
                    )));
                }
                if let Some(current_heading) = current_heading.as_mut() {
                    current_heading.push(Event::Text(pulldown_cmark::CowStr::Boxed(
                        text.clone().into_boxed_str(),
                    )));
                }

                if let Some(current_section) = current_section.as_mut() {
                    current_section.links.push(Link {
                        absolute_url: path_clean::clean(absolute_url),
                        title: Some(title),
                        id: Some(id),
                        text,
                    });
                }
            }
            Event::End(TagEnd::Paragraph) => {
                let paragraph = current_paragraph.take().unwrap();

                let mut text = vec![];
                for p in paragraph {
                    match p {
                        Event::Text(ref t) => text.push(t.to_string()),
                        Event::Code(ref t) => text.push(t.to_string()),
                        _ => unreachable!("expected text"),
                    }
                }

                current_section
                    .as_mut()
                    .unwrap()
                    .content
                    .push(text.join(""));
            }
            Event::End(TagEnd::Heading(_)) => {
                let mut heading = current_heading.take().unwrap();

                let heading = if first_header {
                    first_header = false;

                    let id = match heading.remove(0) {
                        Event::Start(Tag::Heading { id, .. }) => id,
                        _ => unreachable!(),
                    };

                    let mut metadata = HashMap::new();
                    for h in heading {
                        let content = match h {
                            Event::Text(ref text) => text.to_string(),
                            _ => panic!("expected text as child of heading"),
                        };
                        let (key, value) = content.split_once(":").expect(": expected");
                        metadata.insert(key.trim().to_string(), value.trim().to_string());
                    }

                    let id = if let Some(id) = id {
                        id.to_string()
                    } else {
                        metadata.get("id").unwrap_or(&"".to_string()).to_string()
                    };
                    let title = metadata.get("title").unwrap_or(&"".to_string()).to_string();
                    page_heading = Some(PageHeading {
                        id,
                        title,
                        metadata: Default::default(),
                    });

                    continue;
                } else if heading.len() == 2 {
                    // Section heading
                    let level = match heading[0] {
                        Event::Start(Tag::Heading { level, .. }) => level.into(),
                        _ => unreachable!(),
                    };
                    let title = match heading[1] {
                        Event::Text(ref text) => text.to_string(),
                        Event::Code(ref text) => text.to_string(),
                        _ => unreachable!("expected text found {:?} {file_path:?}", heading[1]),
                    };

                    SectionHeading { level, text: title }
                } else if heading.len() >= 3 {
                    let level = match heading[0] {
                        Event::Start(Tag::Heading { level, .. }) => level.into(),
                        _ => unreachable!(),
                    };

                    let mut title = vec![];
                    for h in heading {
                        match h {
                            Event::Text(ref text) => title.push(text.to_string()),
                            Event::Code(ref text) => title.push(text.to_string()),
                            _ => {}
                        }
                    }

                    SectionHeading {
                        level,
                        text: title.join(""),
                    }
                } else {
                    panic!("Invalid heading");
                };

                current_section = Some(Section {
                    heading: Some(heading),
                    content: vec![],
                    links: vec![],
                    codes: vec![],
                });
            }
            _ => {}
        }
    }

    sections.push(current_section.unwrap());

    Page {
        heading: page_heading,
        sections: sections
            .into_iter()
            .filter(|s| !s.content.is_empty() || !s.links.is_empty() || !s.codes.is_empty())
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_visibility() {
        let code = r###"
---
title: Column Visibility Guide
---

## Examples

Want to skip to the implementation? Check out these examples:

- [column-visibility](../../framework/react/examples/column-visibility)
- [column-ordering](../../framework/react/examples/column-ordering)
- [sticky-column-pinning](../../framework/react/examples/column-pinning-sticky)

### Other Examples

- [SolidJS column-visibility](../../framework/solid/examples/column-visibility)
- [Svelte column-visibility](../../framework/svelte/examples/column-visibility)

## Column Visibility Guide

The column visibility feature allows table columns to be hidden or shown dynamically. In previous versions of react-table, this feature was a static property on a column, but in v8, there is a dedicated `columnVisibility` state and APIs for managing column visibility dynamically.

### Column Visibility State

The `columnVisibility` state is a map of column IDs to boolean values. A column will be hidden if its ID is present in the map and the value is `false`. If the column ID is not present in the map, or the value is `true`, the column will be shown.

```jsx
const [columnVisibility, setColumnVisibility] = useState({
  columnId1: true,
  columnId2: false, //hide this column by default
  columnId3: true,
});

const table = useReactTable({
  //...
  state: {
    columnVisibility,
    //...
  },
  onColumnVisibilityChange: setColumnVisibility,
});
```

Alternatively, if you don't need to manage the column visibility state outside of the table, you can still set the initial default column visibility state using the `initialState` option.

> **Note**: If `columnVisibility` is provided to both `initialState` and `state`, the `state` initialization will take precedence and `initialState` will be ignored. Do not provide `columnVisibility` to both `initialState` and `state`, only one or the other.

```jsx
const table = useReactTable({
  //...
  initialState: {
    columnVisibility: {
      columnId1: true,
      columnId2: false, //hide this column by default
      columnId3: true,
    },
    //...
  },
});
```

"###.trim();
        let mut page = parse(code, "/base/path/table/guide/docs".parse().unwrap());

        assert_eq!(
            page.heading,
            Some(PageHeading {
                id: "".to_string(),
                title: "Column Visibility Guide".to_string(),
                metadata: Default::default(),
            })
        );
        assert_eq!(page.sections.len(), 4);

        let section = page.sections.remove(0);

        assert_eq!(
            section.heading,
            Some(SectionHeading {
                level: Level::H2,
                text: "Examples".to_string(),
            })
        );

        assert_eq!(
            section.content,
            vec!["Want to skip to the implementation? Check out these examples:".to_string(),]
        );

        assert_eq!(section.links.len(), 3);
        assert_eq!(
            section.links[0].absolute_url,
            PathBuf::from("/base/path/framework/react/examples/column-visibility")
        );
        assert_eq!(section.links[0].text, "column-visibility");
        assert_eq!(
            section.links[1].absolute_url,
            PathBuf::from("/base/path/framework/react/examples/column-ordering")
        );
        assert_eq!(section.links[1].text, "column-ordering");
        assert_eq!(
            section.links[2].absolute_url,
            PathBuf::from("/base/path/framework/react/examples/column-pinning-sticky")
        );
        assert_eq!(section.links[2].text, "sticky-column-pinning");

        assert_eq!(section.codes.len(), 0);

        let section = page.sections.remove(0);

        assert_eq!(
            section.heading,
            Some(SectionHeading {
                level: Level::H3,
                text: "Other Examples".to_string(),
            })
        );

        assert_eq!(section.content, Vec::<String>::new());

        assert_eq!(section.links.len(), 2);
        assert_eq!(
            section.links[0].absolute_url,
            PathBuf::from("/base/path/framework/solid/examples/column-visibility")
        );
        assert_eq!(section.links[0].text, "SolidJS column-visibility");
        assert_eq!(
            section.links[1].absolute_url,
            PathBuf::from("/base/path/framework/svelte/examples/column-visibility")
        );
        assert_eq!(section.links[1].text, "Svelte column-visibility");

        assert_eq!(section.codes.len(), 0);

        let section = page.sections.remove(0);

        assert_eq!(
            section.heading,
            Some(SectionHeading {
                level: Level::H2,
                text: "Column Visibility Guide".to_string(),
            })
        );

        assert_eq!(section.content, vec![
            "The column visibility feature allows table columns to be hidden or shown dynamically. In previous versions of react-table, this feature was a static property on a column, but in v8, there is a dedicated columnVisibility state and APIs for managing column visibility dynamically.".to_string(),
        ]);

        assert_eq!(section.links.len(), 0);
        assert_eq!(section.codes.len(), 0);

        let section = page.sections.remove(0);

        println!("{:#?}", section);

        assert_eq!(
            section.heading,
            Some(SectionHeading {
                level: Level::H3,
                text: "Column Visibility State".to_string(),
            })
        );

        assert_eq!(section.content, vec![
            "The columnVisibility state is a map of column IDs to boolean values. A column will be hidden if its ID is present in the map and the value is false. If the column ID is not present in the map, or the value is true, the column will be shown.".to_string(),
            "Alternatively, if you don't need to manage the column visibility state outside of the table, you can still set the initial default column visibility state using the initialState option.".to_string(),
            "Note: If columnVisibility is provided to both initialState and state, the state initialization will take precedence and initialState will be ignored. Do not provide columnVisibility to both initialState and state, only one or the other.".to_string(),
        ]);
    }

    #[test]
    fn test_ag_grid() {
        let file_content = r###"
---
title: AG Grid - An alternative enterprise data-grid solution
---

<p>
  <a href="https://ag-grid.com/react-data-grid/?utm_source=reacttable&utm_campaign=githubreacttable">
    <img src="https://blog.ag-grid.com/content/images/2021/02/new-logo-1.png" style={{ width:400 }} />
  </a>
</p>

While we clearly love TanStack Table, we acknowledge that it is not a "batteries" included product packed with customer support and enterprise polish. We realize that some of our users may need this though! To help out here, we want to introduce you to AG Grid, an enterprise-grade data grid solution that can supercharge your applications with its extensive feature set and robust performance. While TanStack Table is also a powerful option for implementing data grids, we believe in providing our users with a diverse range of choices that best fit their specific requirements. AG Grid is one such choice, and we're excited to highlight its capabilities for you.

## Why Choose [AG Grid](https://ag-grid.com/react-data-grid/?utm_source=reacttable&utm_campaign=githubreacttable)?

Here are some good reasons to consider AG Grid for your next project:

### Comprehensive Feature Set

AG Grid offers an extensive set of features, making it a versatile and powerful data grid solution. With AG Grid, you get access to a wide range of functionalities that cater to the needs of complex enterprise applications. From advanced sorting, filtering, and grouping capabilities to column pinning, multi-level headers, and tree data structure support, AG Grid provides you with the tools to create dynamic and interactive data grids that meet your application's unique demands.

### High Performance

When it comes to handling large datasets and achieving exceptional performance, AG Grid delivers outstanding results. It employs highly optimized rendering techniques, efficient data updates, and virtualization to ensure smooth scrolling and fast response times, even when dealing with thousands or millions of rows of data. AG Grid's performance optimizations make it an excellent choice for applications that require high-speed data manipulation and visualization.

### Customization and Extensibility

AG Grid is designed to be highly customizable and extensible, allowing you to tailor the grid to your specific needs. It provides a rich set of APIs and events that enable you to integrate custom functionality seamlessly. You can define custom cell renderers, editors, filters, and aggregators to enhance the grid's behavior and appearance. AG Grid also supports a variety of themes, allowing you to match the grid's visual style to your application's design.

### Support for Enterprise Needs

As an enterprise-focused solution, AG Grid caters to the requirements of complex business applications. It offers enterprise-specific features such as row grouping, column pinning, server-side row model, master/detail grids, and rich editing capabilities. AG Grid also integrates well with other enterprise frameworks and libraries, making it a reliable choice for large-scale projects.

### Active Development and Community Support

AG Grid benefits from active development and a thriving community of developers. The team behind AG Grid consistently introduces new features and enhancements, ensuring that the product evolves to meet the changing needs of the industry. The community support is robust, with forums, documentation, and examples readily available to assist you in utilizing the full potential of AG Grid.

## Conclusion

While TanStack Table remains a powerful and flexible option for implementing data grids, we understand that different projects have different requirements. AG Grid offers a compelling enterprise-grade solution that may be particularly suited to your needs. Its comprehensive feature set, high performance, customization options, and focus on enterprise requirements make AG Grid an excellent choice for projects that demand a robust and scalable data grid solution.

We encourage you to explore AG Grid further by visiting their website and trying out their demo. Remember that both TanStack Table and AG Grid have their unique strengths and considerations. We believe in providing options to our users, empowering you to make informed decisions and choose the best fit for your specific use case.

Visit the [AG Grid website](https://www.ag-grid.com).

"###.trim();
        let page = parse(
            file_content,
            "/base/path/table/enterprise/ag-grid.md".parse().unwrap(),
        );

        assert_eq!(
            page.sections[1].heading,
            Some(SectionHeading {
                level: Level::H2,
                text: "Why Choose AG Grid?".to_string(),
            })
        );

        // The header contains a link. We don't track it yet.
        // TODO: Add support for links in headers
        assert_eq!(page.sections[1].links.len(), 0);

        println!("{:#?}", page);
    }

    #[test]
    fn test_svelte_table() {
        let file_content = r###"
---
title: Svelte Table
---

The `@tanstack/svelte-table` adapter is a wrapper around the core table logic. Most of it's job is related to managing state the "svelte" way, providing types and the rendering implementation of cell/header/footer templates.

## `createSvelteTable`

Takes an `options` object and returns a table.

```svelte
<script>

import { createSvelteTable } from '@tanstack/svelte-table'

const table = createSvelteTable(options)

</script>
```
"###.trim();
        let page = parse(
            file_content,
            "/base/path/table/frameword/svelte/svelte-table.md"
                .parse()
                .unwrap(),
        );

        assert_eq!(
            page.sections[1].heading,
            Some(SectionHeading {
                level: Level::H2,
                text: "createSvelteTable".to_string(),
            })
        );
    }
}
