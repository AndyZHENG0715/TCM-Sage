# Prompt 1: Add Conversation Memory (Full-Stack)

Send to: **full-stack agent**

## Goal
Currently each message is sent independently — the LLM has zero context of prior turns. Add conversation memory so the LLM can reference earlier messages.

## What to change

### Backend: `src/api.py`
- `QueryRequest` model (line 130): add `chat_history: list[dict] = []` field
  - Each item: `{"role": "user"|"assistant", "content": "..."}`
- `generate_sse_stream()` (line 178): pass `chat_history` through to `run_query_stream()`

### Backend: `src/ui_backend.py`
- `run_query_stream()` (line 285): accept `chat_history` param
- Before building `chain_input`, format the chat history into a conversation string and prepend it to the prompt context. Simple approach:
  ```
  history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in chat_history[-6:]])
  ```
  Limit to last 6 messages to avoid token overflow.
- Append `history_text` to the prompt template's context section

### Frontend: `web/lib/api.ts`
- `streamQuery()` (line 17): change signature to accept `(question, chatHistory)`
- Send `{ question, chat_history: chatHistory }` in the POST body

### Frontend: `web/hooks/useChat.ts`
- In `sendMessage()` (line 40): pass current `messages` array to `streamQuery()`
- Map messages to the `{role, content}` format expected by backend

## Verification
1. Ask "什么是阴阳?" → get response
2. Follow up: "能详细说说吗?" → response should reference yin-yang from turn 1
3. Third turn: "之前提到的五行呢?" → should know what "之前" refers to

## Commit
```
feat: add conversation memory with chat history support
```
