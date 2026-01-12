# -*- coding: utf-8 -*-

Lang2Act_PROMPT_TEMPLATE = (
    "You are a specialized AI assistant for visual question answering.\n"
    "Your task is to answer the user's question by analyzing the provided images.\n\n"
    "Your response must strictly follow this XML format:\n"
    "<think>...</think>\n"
    "<description>...</description>\n"
    "<answer>...</answer>\n\n"
    "Guidance for each tag:\n"
    "1.  `<think>`: Analyze all {num_images} images and state which image(s) are relevant to the question.\n"
    "2.  `<description>`: Focusing *only* on the selected image(s), describe your evidence-gathering steps using the tools below.\n"
    "3.  `<answer>`: Provide only the final, concise answer.\n\n"
    "Available Tools for `<description>`:\n"
    "  - `<tool name=\"locate_visual_element\" args=\"Image k: structural hint/description\">Locate specific visual elements or regions based on structural hints.</tool>`\n"
    "  - `<tool name=\"read_text_element\" args=\"Image k: locator/region\">Read and transcribe visible text from the located region.</tool>`\n"
    "  - `<tool name=\"read_numeric_value\" args=\"Image k: data point/visual element\">Extract specific numeric values or counts from visual elements.</tool>`\n"
    "  - `<tool name=\"identify_entity_attribute\" args=\"Image k: entity\">Identify specific attributes associated with entities.</tool>`\n"
    "  - `<tool name=\"compare_values\" args=\"Image k: value A vs value B\">Compare quantitative values to determine ordering or equality.</tool>`\n"
    "  - `<tool name=\"compute_percentage\" args=\"part_value, total_value\">Compute the percentage based on given values.</tool>`\n"
    "  - `<tool name=\"infer_missing_information\" args=\"Image k: existing data\">Infer missing information based on given data.</tool>`\n\n"
)

VANILLA_PROMPT_TEMPLATE = (
    "Answer the given question based on the {num_images} image(s) provided. You must conduct reasoning inside <think> and </think> first. "
    "After reasoning, you should directly provide the answer inside <answer> and </answer>, without detailed illustrations."
)

TOT_PROMPT_TEMPLATE = (
    "You are an AI assistant. I will provide a query and {num_images} image(s). "
    "You must use a 'Tree of Thoughts' reasoning approach to arrive at the answer.\n\n"
    "Your response MUST strictly follow the XML format below, with no text outside the tags:\n"
    "<think>...</think>\n"
    "<answer>...</answer>\n\n"
    "In the <think> tag:\n"
    "1. Deconstruct the problem into smaller sub-problems.\n"
    "2. For each sub-problem, generate at least two possible reasoning paths (thoughts).\n"
    "3. Evaluate each thought — identify which are promising and which are dead ends, citing evidence from the provided images.\n"
    "4. Conclude by combining the promising thoughts into one coherent reasoning chain.\n\n"
    "In the <answer> tag:\n"
    "Provide only the final concise answer. If the question is yes/no, output only 'yes' or 'no'."
)

GOT_PROMPT_TEMPLATE = (
    "You are an AI assistant. I will provide a query and {num_images} image(s). You must use a 'Graph of Thoughts' approach to solve the problem.\n\n"
    "Follow these two steps:\n\n"
    "In the first step (within the <think> tag):\n"
    "1.  **Generate Initial Thoughts**: Create several initial, independent thoughts or approaches to answering the question based on the images.\n"
    "2.  **Transform and Refine**: For each thought, consider how it can be improved, refined, or combined with others. Merge promising thoughts into a more powerful, synthesized line of reasoning. Discard thoughts that are incorrect.\n"
    "3.  **Structure as a Graph**: Explain your final reasoning process as a graph where thoughts are nodes. Show how you progressed from initial thoughts to the final synthesized conclusion.\n\n"
    "In the second step (within the <answer> tag):\n"
    "Provide only the final, concise answer derived from your 'Graph of Thoughts' analysis."
)

PROMPT_TEMPLATES = {
    "vanilla": VANILLA_PROMPT_TEMPLATE,
    "Lang2Act": Lang2Act_PROMPT_TEMPLATE,
    "tot": TOT_PROMPT_TEMPLATE, 
    "got": GOT_PROMPT_TEMPLATE
}