# utils/token_cost_tracker.py

from src.config.model_cost_config import MODEL_COST_CONFIG

# Global dictionary to track usage
USAGE_TRACKER = {}

def calculate_model_cost(model_name, **token_usage):
    """
    Calculate cost for a model based on token usage.
    Cache tokens are subtracted from input tokens before cost calculation.
    """
    if model_name not in MODEL_COST_CONFIG:
        raise ValueError(f"Model '{model_name}' not found in cost config.")

    rates = MODEL_COST_CONFIG[model_name]
    input_tokens = token_usage.get("input", 0)
    output_tokens = token_usage.get("output", 0)
    cache_tokens = token_usage.get("cache", 0)

    # Ensure input tokens are not negative after cache
    net_input_tokens = max(0, input_tokens - cache_tokens)

    cost = {
        "input": (net_input_tokens / 1_000_000) * rates.get("input", 0.0),
        "output": (output_tokens / 1_000_000) * rates.get("output", 0.0),
        "cache": (cache_tokens / 1_000_000) * rates.get("cache", 0.0),
    }

    cost["total"] = sum(cost.values())
    return cost

def record_llm_usage(chat_id, model_name, **token_usage):
    """
    Accumulate token usage by chat_id for later cost aggregation.
    Example: record_usage("chat123", "gpt-4o", input=1000, output=500, cache=100)
    """
    if chat_id not in USAGE_TRACKER:
        USAGE_TRACKER[chat_id] = {}

    if model_name not in USAGE_TRACKER[chat_id]:
        USAGE_TRACKER[chat_id][model_name] = {"input": 0, "output": 0, "cache": 0}

    for key in ["input", "output", "cache"]:
        USAGE_TRACKER[chat_id][model_name][key] += token_usage.get(key, 0)

def get_total_cost(chat_id, clear_after=True):
    """
    Aggregate and calculate total cost for a given chat_id.
    If clear_after=True, clears the tracked usage after calculation.
    """
    if chat_id not in USAGE_TRACKER:
        return {}

    final_costs = {}
    grand_total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_tokens = 0
    total_uncached_tokens = 0


    


    for model_name, usage in USAGE_TRACKER[chat_id].items():
        cost_data = calculate_model_cost(model_name, **usage)

        input_tokens = usage.get("input", 0)
        output_tokens = usage.get("output", 0)
        cache_tokens = usage.get("cache", 0)
        final_costs[model_name] = cost_data
        grand_total_cost += cost_data["total"]

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_cache_tokens += cache_tokens
        total_uncached_tokens += total_input_tokens - total_cache_tokens

    final_costs["summary"] = {
        "grand_total_cost": grand_total_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cache_tokens": total_cache_tokens,
        "total_uncached_tokens": total_uncached_tokens,
    }

    if clear_after:
        del USAGE_TRACKER[chat_id]

    return final_costs


def clear_llm_usage(chat_id):
    """
    Manually clear recorded usage for a chat_id.
    """
    if chat_id in USAGE_TRACKER:
        del USAGE_TRACKER[chat_id]

def get_usage_snapshot():
    """
    Return a snapshot of current usage tracker (for debugging/testing).
    """
    return USAGE_TRACKER.copy()
