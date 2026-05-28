#!/usr/bin/env python3

from qwen_query_template import query

if __name__ == "__main__":
    query(model_name="Qwen/Qwen3.5-9B", polities_to_evaluate=["sc"])