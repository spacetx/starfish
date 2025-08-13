```mermaid
graph LR
    Retriever_Creator["Retriever Creator"]
    Classic_RAG_Retriever["Classic RAG Retriever"]
    Retriever_Creator -- "instantiates" --> Classic_RAG_Retriever
```

[![CodeBoarding](https://img.shields.io/badge/Generated%20by-CodeBoarding-9cf?style=flat-square)](https://github.com/CodeBoarding/GeneratedOnBoardings)[![Demo](https://img.shields.io/badge/Try%20our-Demo-blue?style=flat-square)](https://www.codeboarding.org/demo)[![Contact](https://img.shields.io/badge/Contact%20us%20-%20contact@codeboarding.org-lightgrey?style=flat-square)](mailto:contact@codeboarding.org)

## Details

The Retrieval Module is a core subsystem responsible for efficiently fetching the most relevant document chunks from the Vector Database based on user queries. It acts as the bridge between the user's information need and the contextual data required by the Large Language Model (LLM) for generating responses. Its boundaries encompass the logic for query processing, interaction with the knowledge base, and preparation of retrieved content.

### Retriever Creator
This component serves as a factory for instantiating various retriever implementations. It abstracts the creation process, allowing other parts of the system to obtain a retriever instance (e.g., `Classic RAG Retriever`) without needing to know the specific concrete class or its initialization details. This design promotes modularity, extensibility, and supports the project's "LLM Agnosticism" and "Modularity" architectural biases by enabling easy swapping or addition of different retrieval strategies.


**Related Classes/Methods**:

- <a href="https://github.com/arc53/DocsGPT/blob/main/application/retriever/retriever_creator.py#L1-L9999" target="_blank" rel="noopener noreferrer">`Retriever Creator`:1-9999</a>


### Classic RAG Retriever
This component embodies a concrete and fundamental retrieval strategy within the RAG pipeline. It encapsulates the core logic for transforming a user query into an effective search query, interacting with the Vector Database to retrieve the most relevant document chunks, and preparing these chunks as contextual information for the LLM. It represents the "how" of fetching information in a standard RAG flow.


**Related Classes/Methods**:

- <a href="https://github.com/arc53/DocsGPT/blob/main/application/retriever/classic_rag.py#L1-L9999" target="_blank" rel="noopener noreferrer">`Classic RAG Retriever`:1-9999</a>




### [FAQ](https://github.com/CodeBoarding/GeneratedOnBoardings/tree/main?tab=readme-ov-file#faq)