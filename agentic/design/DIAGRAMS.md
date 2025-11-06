# System Diagrams

This document contains visual diagrams for the UDA-Hub multi-agent system using Mermaid syntax.

---

## 1. High-Level System Architecture

```mermaid
graph TD
    A[External Support System] -->|Ticket| B[UDA-Hub API]
    B --> C[Create Ticket in DB]
    C --> D[Initialize State]
    D --> E[Supervisor Agent]

    E -->|Route| F[Classifier Agent]
    E -->|Route| G[Resolver Agent]
    E -->|Route| H[Tool Agent]
    E -->|Route| I[Escalation Agent]

    F -->|Classification| E
    G -->|Resolution| E
    H -->|Tool Results| E
    I -->|Escalation Summary| J[END]

    G -->|Query| K[RAG System]
    K -->|Articles| G

    H -->|Query| L[CultPass DB]
    L -->|Data| H

    E -->|Resolved| J

    style E fill:#ff9999
    style F fill:#99ccff
    style G fill:#99ff99
    style H fill:#ffcc99
    style I fill:#ff99ff
    style K fill:#ffff99
    style L fill:#cccccc
```

---

## 2. Supervisor Routing Flow

```mermaid
flowchart TD
    START([START]) --> SUP1[Supervisor: Analyze State]

    SUP1 --> CHECK1{Classification\\nExists?}
    CHECK1 -->|No| CLASS[Classifier Agent]
    CHECK1 -->|Yes| CHECK2{Resolution\\nAttempted?}

    CLASS --> SUP2[Supervisor: Review Classification]
    SUP2 --> CHECK2

    CHECK2 -->|No| RES[Resolver Agent]
    CHECK2 -->|Yes| CHECK3{Confidence\\n>= 0.7?}

    RES --> CHECK4{Needs\\nData?}
    CHECK4 -->|Yes| TOOL[Tool Agent]
    CHECK4 -->|No| SUP3[Supervisor: Review Resolution]

    TOOL --> SUP4[Supervisor: Data Retrieved]
    SUP4 --> RES

    SUP3 --> CHECK3

    CHECK3 -->|Yes| RESOLVED([END: Resolved])
    CHECK3 -->|No| ESC[Escalation Agent]

    ESC --> ESCALATED([END: Escalated])

    style SUP1 fill:#ff9999
    style SUP2 fill:#ff9999
    style SUP3 fill:#ff9999
    style SUP4 fill:#ff9999
    style CLASS fill:#99ccff
    style RES fill:#99ff99
    style TOOL fill:#ffcc99
    style ESC fill:#ff99ff
```

---

## 3. Agent Interaction Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant S as Supervisor
    participant C as Classifier
    participant R as Resolver
    participant RAG as RAG System
    participant T as Tool Agent
    participant DB as CultPass DB
    participant E as Escalation

    U->>S: Submit Ticket: "Can't login"
    S->>C: Route to Classifier
    C->>C: Analyze ticket content
    C->>S: Return: type=technical, urgency=high

    S->>R: Route to Resolver
    R->>RAG: Query: "login issues"
    RAG->>RAG: Semantic search
    RAG->>R: Return: Login article
    R->>R: Generate answer (confidence=0.85)
    R->>S: Return: Resolved with answer

    S->>U: Response: "Try 'Forgot Password'..."

    Note over S,U: Alternative: Low Confidence Flow

    U->>S: Submit Ticket: "Refund request"
    S->>C: Route to Classifier
    C->>S: Return: type=billing, urgency=medium

    S->>R: Route to Resolver
    R->>RAG: Query: "refund policy"
    RAG->>R: Return: No refunds without approval
    R->>R: Generate answer (confidence=0.45)
    R->>S: Return: Low confidence

    S->>E: Route to Escalation
    E->>E: Create summary
    E->>S: Return: Escalation summary
    S->>U: Response: "Escalated to human agent"
```

---

## 4. RAG System Architecture

```mermaid
graph LR
    A[Knowledge Articles JSONL] --> B[Load from Database]
    B --> C[Create Documents]
    C --> D[Generate Embeddings]
    D --> E[InMemoryVectorStore]

    F[User Query] --> G[Embed Query]
    G --> H[Similarity Search]
    E --> H
    H --> I[Top-K Articles]
    I --> J[Resolver Agent]
    J --> K[Generate Answer]
    K --> L[Calculate Confidence]
    L --> M{Confidence\\n>= 0.7?}
    M -->|Yes| N[Return Answer]
    M -->|No| O[Escalate]

    style E fill:#ffff99
    style J fill:#99ff99
    style M fill:#ff9999
```

---

## 5. Memory Architecture

```mermaid
graph TD
    subgraph "Short-Term Memory (Session)"
        A[Ticket Created] --> B[thread_id = ticket_id]
        B --> C[MemorySaver Checkpointer]
        C --> D[State Persisted After Each Node]
        D --> E[Multi-Turn Conversation]
        E --> C
    end

    subgraph "Long-Term Memory (Cross-Session)"
        F[Ticket Resolved] --> G[Extract Key Info]
        G --> H[Store in Database]
        H --> I[Generate Embedding]
        I --> J[Vector Store]
        J --> K[Semantic Search]
        K --> L[Retrieve Past Issues]
    end

    E --> M[Ticket Closed]
    M --> F

    L --> N[Use in Current Resolution]

    style C fill:#99ccff
    style J fill:#ffff99
```

---

## 6. State Transition Diagram

```mermaid
stateDiagram-v2
    [*] --> new: Ticket Created
    new --> classifying: Supervisor Routes to Classifier
    classifying --> classified: Classification Complete
    classified --> resolving: Supervisor Routes to Resolver
    resolving --> resolving: Tool Agent Provides Data
    resolving --> resolved: High Confidence (>= 0.7)
    resolving --> escalated: Low Confidence (< 0.7)
    resolved --> [*]: Ticket Closed
    escalated --> [*]: Human Agent Assigned

    note right of classified
        Classification stored in
        TicketMetadata table
    end note

    note right of resolved
        Resolution stored in
        TicketMessage table
    end note
```

---

## 7. Tool Agent Operations

```mermaid
graph TD
    A[Tool Agent Invoked] --> B{Which Tool?}

    B -->|User Lookup| C[user_lookup_tool]
    B -->|Subscription| D[subscription_management_tool]
    B -->|Experience Search| E[experience_search_tool]
    B -->|Reservation| F[reservation_management_tool]
    B -->|Refund| G[refund_processing_tool]

    C --> H[Query CultPass DB]
    D --> H
    E --> H
    F --> H
    G --> H

    H --> I{Success?}
    I -->|Yes| J[Return Structured Result]
    I -->|No| K[Return Error Message]

    J --> L[Update State]
    K --> L
    L --> M[Return to Supervisor]

    style H fill:#cccccc
    style I fill:#ff9999
```

---

## 8. Confidence Scoring Flow

```mermaid
flowchart LR
    A[Retrieved Articles] --> B[Calculate Similarity Score]
    C[Generated Answer] --> D[Calculate Completeness Score]
    E[Article Quality] --> F[Calculate Quality Score]
    G[Classification Match] --> H[Calculate Context Score]

    B --> I[Weight: 0.4]
    D --> J[Weight: 0.3]
    F --> K[Weight: 0.2]
    H --> L[Weight: 0.1]

    I --> M[Sum Weighted Scores]
    J --> M
    K --> M
    L --> M

    M --> N[Final Confidence]
    N --> O{Threshold?}

    O -->|>= 0.8| P[High: Resolve]
    O -->|0.7-0.79| Q[Medium: Resolve with Disclaimer]
    O -->|0.5-0.69| R[Low: Escalate with Summary]
    O -->|< 0.5| S[Very Low: Escalate Immediately]

    style N fill:#ffff99
    style O fill:#ff9999
```

---

## 9. Database Schema Overview

```mermaid
erDiagram
    Account ||--o{ User : has
    Account ||--o{ Ticket : has
    Account ||--o{ Knowledge : has
    User ||--o{ Ticket : creates
    Ticket ||--|| TicketMetadata : has
    Ticket ||--o{ TicketMessage : contains

    Account {
        string account_id PK
        string account_name
        datetime created_at
    }

    User {
        string user_id PK
        string account_id FK
        string external_user_id
        string user_name
    }

    Ticket {
        string ticket_id PK
        string account_id FK
        string user_id FK
        string channel
        datetime created_at
    }

    TicketMetadata {
        string ticket_id PK_FK
        string status
        string main_issue_type
        string tags
    }

    TicketMessage {
        string message_id PK
        string ticket_id FK
        enum role
        text content
        datetime created_at
    }

    Knowledge {
        string article_id PK
        string account_id FK
        string title
        text content
        string tags
    }
```

---

## 10. Error Handling Flow

```mermaid
flowchart TD
    A[Agent Execution] --> B{Error\\nOccurred?}
    B -->|No| C[Continue Normal Flow]
    B -->|Yes| D{Error Type?}

    D -->|RAG Failure| E[Log Error]
    D -->|Tool Failure| F[Retry Once]
    D -->|LLM Timeout| G[Retry with Shorter Context]
    D -->|Invalid Data| H[Use Default Values]

    E --> I[Route to Escalation]
    F --> J{Retry\\nSuccess?}
    G --> K{Retry\\nSuccess?}
    H --> C

    J -->|Yes| C
    J -->|No| I
    K -->|Yes| C
    K -->|No| I

    I --> L[Escalation Agent]
    L --> M[Create Error Summary]
    M --> N[END: Escalated]

    style D fill:#ff9999
    style I fill:#ff99ff
```

---

## 11. Multi-Turn Conversation Flow

```mermaid
sequenceDiagram
    participant U as User
    participant S as Supervisor
    participant C as Checkpointer
    participant R as Resolver

    Note over U,R: Initial Message
    U->>S: "How do I reset password?"
    S->>C: Save state (thread_id=ticket_123)
    S->>R: Route to Resolver
    R->>R: Use RAG, generate answer
    R->>S: Return answer (confidence=0.85)
    S->>C: Save updated state
    S->>U: "Tap 'Forgot Password'..."

    Note over U,R: Follow-up Message (same thread_id)
    U->>S: "I didn't receive the email"
    S->>C: Load state (thread_id=ticket_123)
    C->>S: Previous messages + state
    S->>R: Route to Resolver with full context
    R->>R: Consider previous answer
    R->>S: Return follow-up answer
    S->>C: Save updated state
    S->>U: "Check spam folder..."
```

---

## 12. Ticket Lifecycle

```mermaid
graph TD
    A[Ticket Created] --> B[Ingested to System]
    B --> C[Supervisor: Initial Analysis]
    C --> D[Classifier: Categorize]
    D --> E{Classification\\nComplete?}
    E -->|Yes| F[Resolver: Attempt Resolution]
    E -->|No| G[Escalate: Invalid Ticket]

    F --> H{RAG\\nFound Articles?}
    H -->|Yes| I[Generate Answer]
    H -->|No| J[Escalate: No Knowledge]

    I --> K{Confidence\\n>= 0.7?}
    K -->|Yes| L[Mark Resolved]
    K -->|No| M[Escalate: Low Confidence]

    F --> N{Needs\\nUser Data?}
    N -->|Yes| O[Tool Agent: Query DB]
    O --> I

    L --> P[Store Resolution]
    P --> Q[Notify User]
    Q --> R[Close Ticket]

    G --> S[Human Agent Queue]
    J --> S
    M --> S
    S --> T[Human Reviews]
    T --> U[Close Ticket]

    style L fill:#99ff99
    style S fill:#ff99ff
```

---

## 13. Knowledge Base Update Flow

```mermaid
flowchart TD
    A[New Article Created] --> B[Load into Knowledge Table]
    B --> C[Generate Embedding]
    C --> D[Add to Vector Store]
    D --> E{Vector Store\\nType?}

    E -->|InMemory| F[Rebuild Vector Store]
    E -->|Persistent| G[Add Document]

    F --> H[Restart Required]
    G --> I[Hot Reload]

    H --> J[Articles Available]
    I --> J

    J --> K[Resolver Can Use]

    style C fill:#ffff99
    style J fill:#99ff99
```

---

## 14. CultPass Database Integration

```mermaid
graph TD
    subgraph "UDA-Hub Core"
        A[Ticket] --> B[TicketMetadata]
        A --> C[TicketMessage]
    end

    subgraph "Tool Agent"
        D[Tool Selection]
        E[user_lookup_tool]
        F[subscription_management_tool]
        G[experience_search_tool]
        H[reservation_management_tool]
    end

    subgraph "CultPass DB"
        I[User]
        J[Subscription]
        K[Experience]
        L[Reservation]
    end

    A --> D
    D --> E
    D --> F
    D --> G
    D --> H

    E --> I
    F --> J
    G --> K
    H --> L

    I --> M[Return to Tool Agent]
    J --> M
    K --> M
    L --> M

    M --> N[Update State]
    N --> A

    style D fill:#ffcc99
    style I fill:#cccccc
    style J fill:#cccccc
    style K fill:#cccccc
    style L fill:#cccccc
```

---

## 15. Escalation Priority Assignment

```mermaid
flowchart TD
    A[Escalation Triggered] --> B[Analyze Ticket Context]

    B --> C{Issue\\nType?}

    C -->|Account Security| D[P1: Critical]
    C -->|Service Outage| D
    C -->|Payment Failure| D

    C -->|Cannot Use Service| E[P2: High]
    C -->|Event in 24h| E
    C -->|Billing Dispute| E

    C -->|Feature Bug| F[P3: Medium]
    C -->|General Support| F

    C -->|Feature Request| G[P4: Low]
    C -->|General Inquiry| G

    D --> H[Immediate Response Required]
    E --> I[Response Within 2 Hours]
    F --> J[Response Within 1 Day]
    G --> K[Response Within 3-5 Days]

    H --> L[Assign to Human Agent]
    I --> L
    J --> L
    K --> L

    style D fill:#ff0000
    style E fill:#ff9900
    style F fill:#ffff00
    style G fill:#99ff99
```

---

## Usage Notes

- **Diagram 1**: Shows overall system architecture and component relationships
- **Diagram 2**: Details supervisor decision logic and routing flow
- **Diagram 3**: Illustrates agent interactions in sequence for typical scenarios
- **Diagram 4**: Explains RAG system components and flow
- **Diagram 5**: Shows memory architecture for both short-term and long-term storage
- **Diagram 6**: State machine for ticket status transitions
- **Diagram 7**: Tool agent operations and database interactions
- **Diagram 8**: Confidence scoring calculation and thresholds
- **Diagram 9**: Database schema for UDA-Hub core system
- **Diagram 10**: Error handling and recovery strategies
- **Diagram 11**: Multi-turn conversation state management
- **Diagram 12**: Complete ticket lifecycle from creation to closure
- **Diagram 13**: Knowledge base update and hot reload process
- **Diagram 14**: Integration between UDA-Hub and CultPass databases
- **Diagram 15**: Escalation priority assignment logic

These diagrams should be referenced alongside the detailed documentation in other design files.

---

## Related Documentation

- **System Overview**: See `ARCHITECTURE.md`
- **Agent Details**: See `AGENT_SPECIFICATIONS.md`
- **Data Flow**: See `DATA_FLOW.md`
- **Memory**: See `MEMORY_STRATEGY.md`
- **RAG**: See `RAG_IMPLEMENTATION.md`
- **Index**: See `README.md`
