# Step 1 — CLI Entry + Chat Template

End-to-end view of what happens between the user typing `./bin/llm "..."` and the wrapped token-ID list arriving at step 2 (embeddings).

```mermaid
flowchart TD
    user[/"User shell:<br/>./bin/llm [flags] prompt"/]

    subgraph main["main.cpp (CLI harness)"]
        direction TB
        parse["Parse argv<br/>--max-tokens N<br/>--prompt P (repeatable)<br/>--interactive<br/>positional prompt"]
        validate{"Validate combo<br/>(reject mixes,<br/>reject unequal<br/>batch lengths)"}
        select{"Pick forward-pass path"}

        path1["Path 1<br/>generate_next_token<br/>1 prompt, 1 token<br/>no GPU residency"]
        path2["Path 2<br/>generate_tokens_resident (single)<br/>1 prompt, N tokens<br/>resident weights + KV cache"]
        path3["Path 3<br/>generate_tokens_resident (batched)<br/>B prompts in lockstep"]

        parse --> validate
        validate -->|invalid| err[/"stderr error,<br/>exit 1"/]
        validate -->|valid| select
        select -->|"prompts==1<br/>max_tokens==1"| path1
        select -->|"prompts==1<br/>max_tokens>1"| path2
        select -->|"prompts>1"| path3
    end

    subgraph chat["apply_chat_template (src/inference.cu:52)"]
        direction TB
        encode["BPETokenizer.encode(prompt)<br/>text -> ~6 token IDs<br/>e.g. 'What is 2+2?'"]
        wrap["Prepend/append special tokens:<br/>128000 begin_of_text<br/>128006 start_header_id<br/>882 'user'<br/>128007 end_header_id<br/>271 '\\n\\n'<br/>... prompt tokens ...<br/>128009 eot_id<br/>128006 start_header_id<br/>78191 'assistant'<br/>128007 end_header_id<br/>271 '\\n\\n'"]
        ids[("std::vector&lt;int&gt;<br/>~17 token IDs<br/>cursor parked after<br/>'assistant\\n\\n'")]

        encode --> wrap --> ids
    end

    next[/"Step 2: Embeddings<br/>(GPU lookup,<br/>4096 floats per token)"/]

    user --> parse
    path1 --> encode
    path2 --> encode
    path3 --> encode
    ids --> next

    classDef entry fill:#e1f5fe,stroke:#0277bd
    classDef decision fill:#fff3e0,stroke:#ef6c00
    classDef path fill:#ede7f6,stroke:#5e35b1
    classDef errnode fill:#ffcdd2,stroke:#c62828
    classDef out fill:#c8e6c9,stroke:#2e7d32

    class user,next entry
    class validate,select decision
    class path1,path2,path3 path
    class err errnode
    class ids out
```

## Sequence view (Path 2: `--max-tokens 5 "What is 2+2?"`)

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant M as main.cpp
    participant W as ModelWeights<br/>(host + device)
    participant G as generate_tokens_resident
    participant T as apply_chat_template
    participant B as BPETokenizer
    participant E as Step 2:<br/>Embeddings

    U->>M: ./bin/llm --max-tokens 5 "What is 2+2?"
    M->>M: Parse argv -> max_tokens=5, 1 prompt
    M->>M: Validate (ok)
    M->>M: Select Path 2 (single, multi-token)
    M->>W: Load BF16 weights to host + GPU (~165s cold)
    M->>G: generate_tokens_resident(weights, prompt, 5)
    G->>T: apply_chat_template(tok, "What is 2+2?")
    T->>B: encode("What is 2+2?")
    B-->>T: ~6 token IDs
    T->>T: Wrap with begin_of_text, user header, eot, assistant header
    T-->>G: ~17 token IDs
    G->>E: hand off integer list
    Note over E: Pipeline continues:<br/>embed -> 32 decoder layers -><br/>final norm -> lm_head -> argmax
```
