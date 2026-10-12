"""Framework integrations. Import the submodule you need (each needs its extra,
e.g. ``pip install "tidevec[haystack]"``):

    from tidevec.integrations.langchain import TideVecVectorStore
    from tidevec.integrations.llamaindex import TideVecVectorStore
    from tidevec.integrations.haystack import TideVecDocumentStore, TideVecEmbeddingRetriever
    from tidevec.integrations.dspy import TideVecRetriever
    from tidevec.integrations.langgraph import TideVecStore, memory_tools
    from tidevec.integrations.crewai import TideVecStorage

All of them accept a ``TideVec`` server client or an embedded ``Local``
database. The MCP server is the separate ``tidevec-mcp`` command
(``tidevec.mcp_server``); TypeScript integrations live in ``sdk/typescript``.
"""
