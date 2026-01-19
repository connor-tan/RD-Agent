# Log Reader

This folder contains a small standalone reader to parse RD-Agent log traces under `./log` and emit structured output similar to the UI.

Planned script (to be added): `read_trace.py`

Intended usage examples:

- Read a trace by folder name under `./log`:

  ```bash
  python app/log_reader/read_trace.py --log_name 2026-01-12_06-38-12-202307
  ```

- Read a trace by absolute/relative path:

  ```bash
  python app/log_reader/read_trace.py --log_dir ./log/2026-01-12_06-38-12-202307
  ```

- Write output to a file:

  ```bash
  python app/log_reader/read_trace.py --log_name 2026-01-12_06-38-12-202307 --out ./trace.json
  ```

Output schema (planned):

```json
{
  "data": {"...": "..."},
  "llm_data": {"...": "..."},
  "token_costs": {"...": "..."}
}
```
