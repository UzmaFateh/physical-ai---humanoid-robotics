# Data Model: Robotics Textbook Structure

This document defines the content structure for the "Physical AI & Humanoid Robotics" textbook. The model is based on a hierarchy of Modules, Chapters, and Content Blocks.

## Entity Relationship Diagram (Conceptual)

```
[Module] 1..* --contains-- 1..* [Chapter]
  |
  +-- [Title]
  +-- [Chapter List]

[Chapter]
  |
  +-- [Title]
  +-- [Learning Objectives]
  +-- 1..* [ContentBlock]

[ContentBlock]
  |
  +-- (is a) --> [Explanation]
  +-- (is a) --> [CodeSnippet]
  +-- (is a) --> [LabAssignment]
```

## Entity Definitions

### Module

Represents a major section of the textbook. There are exactly four modules.

| Field | Type | Description | Constraints |
|---|---|---|---|
| `id` | String | A unique identifier for the module (e.g., `module1-ros2`). | Required, unique. |
| `title` | String | The full title of the module. | Required. |
| `chapters` | Array<Chapter> | An ordered list of chapters within the module. | Must contain 3-4 chapters. |

### Chapter

Represents a single chapter within a module.

| Field | Type | Description | Constraints |
|---|---|---|---|
| `id` | String | A unique identifier for the chapter (e.g., `ch01-intro-to-ros2`). | Required, unique. |
| `title` | String | The full title of the chapter. | Required. |
| `learning_objectives` | Array<String> | A list of key skills the student will acquire. | Required, Min 1 item. |
| `content_blocks` | Array<ContentBlock> | An ordered list of content blocks that make up the chapter. | Required. |

### ContentBlock (Interface)

Represents a generic block of content within a chapter. This is an abstract type implemented by `Explanation`, `CodeSnippet`, and `LabAssignment`.

| Field | Type | Description |
|---|---|---|
| `type` | Enum | The type of content block (`EXPLANATION`, `CODE`, `LAB`). |

### Explanation (implements ContentBlock)

A block of text explaining a concept.

| Field | Type | Description |
|---|---|---|
| `content` | Markdown | The textual explanation of a topic. |

### CodeSnippet (implements ContentBlock)

A block of code demonstrating a concept.

| Field | Type | Description |
|---|---|---|
| `language` | String | The programming language of the snippet (e.g., `python`, `xml`, `bash`). |
| `code` | String | The raw code content. |
| `description` | Markdown | A brief explanation of what the code does. |
| `is_executable` | Boolean | If `true`, the code is intended to be run by the student. |

### LabAssignment (implements ContentBlock)

A practical exercise for the student to complete.

| Field | Type | Description |
|---|---|---|
| `title` | String | The title of the lab assignment. |
| `problem_statement` | Markdown | A description of the task the student needs to accomplish. |
| `expected_outcome` | Markdown | A description of the successful result. |
| `hints` | Array<String> | Optional hints to guide the student. |

## State Transitions

N/A for this content-based model. The content is static once generated.
