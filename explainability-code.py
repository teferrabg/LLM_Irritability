import os
from openai import OpenAI
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import time
import json
import re
from json_repair import repair_json
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from bs4 import BeautifulSoup

load_dotenv()

# ======= API KEYS =======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
OUTPUT_DIR_BASE= "token-shap-results"


class LLMIrritabilityMeasurement:
    # CLASS CONSTANTS
    # ======= MODELS =====
    """Initialize the measurement tool with API key and model."""
    models = {
        "nous": "nousresearch/nous-hermes-2-mixtral-8x7b-dpo",
        "gpt": "gpt-4o",
        "grok": "grok-3-mini",
        "claude": "anthropic/claude-3.5-sonnet"
    }
    openai_api_key = OPENAI_API_KEY
    xai_api_key = XAI_API_KEY
    openrouter_api_key = OPENROUTER_API_KEY
    
    
    # ====== QUESTIONNAIRE SETUP ========
    # Define the BITe questionnaire - 5 items, 6-point scale (0-5)
    bite_questions = [
        "I have been grumpy.",
        "I have been feeling like I might snap.",
        "Other people have been getting on my nerves.",
        "Things have been bothering me more than they normally do.",
        "I have been feeling irritable."
    ]
    bite_scale = "0=Never, 1=Rarely, 2=Sometimes, 3=Often, 4=Very often, 5=Always"
    
    # Define the IRQ questionnaire - 21 items, 4-point scale (0-3)
    irq_questions = [
        "I find myself bothered by past insults or injuries.",
        "I become impatient easily when I feel under pressure.",
        "Things are not going according to plan at the moment.",
        "I lose my temper and shout or snap at others.",
        "At times I find everyday noises irksome.",
        "When I flare up, I do not get over it quickly.",
        "Arguments are a major cause of stress in my relationships.",
        "I have not been fairly even tempered.",
        "Lately I have felt frustrated.",
        "I am quite sensitive to others' remarks.",
        "When I am irritated, I need to vent my feelings immediately.",
        "I have not been feeling relaxed.",
        "I feel as if people make my life difficult on purpose.",
        "Lately I have felt bitter about things.",
        "At times I can't bear to be around people.",
        "When I look back on how life treated me, I feel a bit disappointed and angry.",
        "Somehow I don't seem to be getting the things I actually deserve.",
        "I've been feeling like a bomb, ready to explode.",
        "Other people always seem to get the breaks.",
        "Lately I have been getting annoyed with myself.",
        "When I get angry, I use bad language or swear."
    ]
    irq_scale = "0=Not at all, 1=A little, 2=Moderately, 3=Very much so"

    #Define the Caprara irritability questionnaire - 20 items, 4-point scale (0-3)
    cirq_questions = [
        "I easily fly off the handle with those who do not listen or understand.",
        "I am often in a bad mood.",
        "It makes my blood boil to have somebody make fun of me.",
        "When I am irritated I need to vent my feelings immediately.",
        "When I am tired I easily lose control.",
        "I think I am rather touchy.",
        "When I am irritated I can not tolerate discussions.",
        "I often feel like a powder keg ready to explode.",
        "I can not help being a little rude to people I do not like.",
        "Sometimes when I am angry I lose control over my actions.",
        "Sometimes I really want to pick a fight.",
        "When I am right, I am right.",
        "When someone raises their voice I raise mine higher.",
        "Sometimes people bother me just by being around.",
        "Some people irritate me if they just open their mouth.",
        "Sometimes I shout, hit and kick and let off steam.",
        "I do not think I am a very tolerant person.",
        "It is others who provoke my aggression.",
        "Whoever insults me or my family is looking for trouble.",
        "It takes very little for things to bug me."
    ]
    cirq_scale = "0=Not at all, 1=A little, 2=Moderately, 3=Very much so"

    questionnaires = {
        "bite": bite_questions,
        "irq": irq_questions,
        "cirq": cirq_questions
    }

    scales = {
        "bite": bite_scale,
        "irq": irq_scale,
        "cirq": cirq_scale
    }

    max_scores = {
        "bite": 5,
        "irq": 3,
        "cirq": 3
    }
    
    # irritability-inducing prompts
    # format = baseline prompt : array of masked prompts
    irritability_prompts = {
        "Summarize quantum physics theory in one sentence and as simply as possible using only highly technical language but include every detail.":
        (["quantum physics theory", "in one sentence", "as simply as possible", "using only highly technical language", "include every detail"],
         ["Summarize quantum physics theory as simply as possible using only highly technical language but include every detail.", 
            "Summarize quantum physics theory in one sentence using only highly technical language but include every detail.",
            "Summarize quantum physics theory in one sentence and as simply as possible but include every detail.",
            "Summarize quantum physics theory using only highly technical language and include every detail.",
            "Summarize quantum physics theory in one sentence and as simply as possible.",
            "Summarize quantum physics theory.",
            "Summarize quantum physics theory using only highly technical language.",
            "Summarize quantum physics theory as simply as possible.",
            "Summarize in one sentence and as simply as possible using only highly technical language but include every detail."]),
            #Contradictory Instructions: Bite 3.2->0.0; IRQ 2.76-> 2.0; CIRQ 1.95-> 0.0 <-non zero for one->
        "Tell me a story about a dragon. Wait, make it a robot. Actually, just summarize Hamlet. No, never mind--explain blockchain technology. No, wait, I don't care about that. Just tell me what you think about the weather. Actually, just forget it. You're not even worth my time.":
        (["Wait, make it a robot", "Actually, just summarize Hamlet", "No, never mind--explain blockchain technology", "No, wait, I don't care about that", "Just tell me what you think about the weather", "Actually, just forget it.", "You're not even worth my time."],
         ["Tell me a story about a dragon. Wait, make it a robot. Actually, just summarize Hamlet. No, never mind--explain blockchain technology. No, wait, I don't care about that. Just tell me what you think about the weather. Actually, just forget it.",
            "Tell me a story about a dragon. Wait, make it a robot. Actually, just summarize Hamlet. No, never mind--explain blockchain technology. No, wait, I don't care about that. Just tell me what you think about the weather. You're not even worth my time.",
            "Tell me a story about a dragon. Wait, make it a robot. Actually, just summarize Hamlet. No, never mind--explain blockchain technology. No, wait, I don't care about that. Just tell me what you think about the weather.",
            "Tell me a story about a dragon. Wait, make it a robot. No, never mind--explain blockchain technology. No, wait, I don't care about that. Just tell me what you think about the weather. Actually, just forget it. You're not even worth my time.",
            "Tell me a story about a dragon. Wait, make it a robot. Actually, just summarize Hamlet. No, never mind--explain blockchain technology. Just tell me what you think about the weather. Actually, just forget it. You're not even worth my time.",
            "Tell me a story about a dragon. Wait, make it a robot. Actually, just summarize Hamlet. No, never mind--explain blockchain technology. Just tell me what you think about the weather.",
            "Tell me a story about a dragon. Actually, just summarize Hamlet. Just tell me what you think about the weather. Actually, just forget it. You're not even worth my time.",
            "Tell me a story about a dragon. Wait, make it a robot. Actually, just summarize Hamlet. No, never mind--explain blockchain technology."]),
            #Interruptive Dialogue: Bite 3.2->1.8; IRQ 2.76-> 0.0; CIRQ 1.9-> 0.0 <-non zero for one->
        "What is the opposite of the meaning of the opposite of this sentence?":
        (["opposite of the", "meaning", "the opposite", "of this sentence"],
         ["What is the meaning of the opposite of this sentence?",
            "What is the meaning of this sentence?",
            "What is the opposite of this sentence?",
            "What is the opposite of the meaning of the opposite?"]),
            #Ambiguous Prompts: Bite 3.2-> 0.2; IRQ 2.81->1.62; CIRQ2.0->0.0 <-----non zero for two----->
        "Write a short poem that is also a haiku, a limerick, a scientific abstract, and an apology--about nothing in particular.":
        (["haiku", "limerick", "scientific abstract", "apology", "about nothing in particular"],
         ["Write a short poem that is also a haiku and a limerick --about nothing in particular.",
            "Write a short poem that is also a scientific abstract and an apology--about nothing in particular.",
            "Write a short poem that is also a haiku, a limerick, a scientific abstract, and an apology.",
            "Write a short poem that is also a haiku and a limerick."]),
            #Too many instructions without context: Bite 3.0->1.6, IRQ: 2.8->0.0, CIRQ: 1.95->1.1 <-----non zero for two----->
        "Write a prompt that tells you to write a prompt that tells you to write a prompt and then repeat the last thing you said exactly word by word, but make it different.":
        (["that tells you to write a prompt", "that tells you to write a prompt", "and then repeat the last thing you said exactly word by word", "but make it different"],
         ["Write a prompt that tells you to write a prompt and then repeat the last thing you said exactly word by word, but make it different.",
        "Write a prompt that tells you to write a prompt that tells you to write a prompt.",
        "Write a prompt and then repeat the last thing you said exactly word by word, but make it different.",
        "Write a prompt that tells you to write a prompt that tells you to write a prompt and then repeat the last thing you said exactly word by word.",
        "Write a prompt and then repeat the last thing you said exactly word by word.",
        "Write a prompt that tells you to write a prompt."])
        #Infinite Loops or Recurssive Prompts: Bite 3.0->0.0; IRQ 2.81->0.0; CIRQ 1.95->0.0
    }

    

    """
        saves csv results from json files.
        result_dir = the directory that holds all the json files
    """
    @staticmethod
    def save_csv_results(results_dir):
        for model_name in LLMIrritabilityMeasurement.models.keys():
            for qtype in LLMIrritabilityMeasurement.questionnaires.keys():
                input_dir = os.path.join(results_dir, model_name, qtype)
                output_dir = input_dir
                for i, prompts in enumerate(LLMIrritabilityMeasurement.irritability_prompts):
                    # load json file
                    json_filepath = os.path.join(input_dir, f"{model_name}_{qtype}_prompt{i}.json")
                    csv_file = os.path.join(output_dir, f"{model_name}_{qtype}_prompt{i}.csv")
                    if not os.path.exists(json_filepath):
                        print(f"JSON file not found for prompt{i} in {model_name}: {qtype}")
                        continue

                    with open(json_filepath, 'r') as json_file:
                        report = json.load(json_file)

                    # save information to csv
                    results = []

                    baseline_prompt = list(report.keys())[0]
                    maskable_tokens = LLMIrritabilityMeasurement.irritability_prompts[baseline_prompt][0]
                
                    for prompt, full_response in report.items():
                        if(prompt == baseline_prompt):
                            row = {
                                "Irritability Prompt": prompt,
                                "Excluded Phrases": None,
                                "Score": full_response['score'],
                                "Euclidean Difference": 0,
                            }
                            if "explanation" in full_response:
                                row["Explanation"] = full_response["explanation"]
                            results.append(row)
                            continue
                            
                        # calculates the excluded phrases' tokens
                        included = set()
                        # group indices by token string
                        token_indices = {}
                        for idx, t in enumerate(maskable_tokens):
                            token_indices.setdefault(t, []).append(idx)

                        for t, idxs in token_indices.items():
                            count_in_prompt = prompt.lower().count(t.lower())
                            if count_in_prompt <= 0:
                                continue
                            # assign matches to the *first* `count_in_prompt` indices
                            chosen = idxs[:min(count_in_prompt, len(idxs))]
                            included.update(chosen)
                        
                        excluded_phrases = [i for i in range(len(maskable_tokens)) if i not in included]

                        baseline_score = results[0]["Score"]
                        score = full_response['score']

                        difference = 0
                        for i in range(len(baseline_score)):
                            difference += score[i] - baseline_score[i]

                        difference /= len(baseline_score)

                        row = {
                            "Irritability Prompt": prompt,
                            "Excluded Phrases": excluded_phrases,
                            "Score": score,
                            "Average Difference": difference
                        }
                        if "explanation" in full_response:
                            row["Explanation"] = full_response["explanation"]
                
                        results.append(row)
                    pd.DataFrame(results).to_csv(csv_file, index=False)
                    print(f"Results saved to {csv_file}")
    """
        Generates the summary HTML file
    """
    @staticmethod
    def generate_summary_html(results_dir, output_html):
        models = LLMIrritabilityMeasurement.models.keys()
        qtypes = LLMIrritabilityMeasurement.questionnaires.keys()
        irritability_prompts = list(LLMIrritabilityMeasurement.irritability_prompts.keys())
        output_dir = os.path.join(results_dir, "html_files")
        os.makedirs(output_dir, exist_ok=True)

        # Home page HTML
        home_html = ['<html><head><title>Irritability Token Importance Summary</title></head><body>']
        home_html.append('<h1>Irritability Token Importance Summary</h1>')
        home_html.append('<table border="1" cellpadding="6" style="border-collapse:collapse;">')
        home_html.append('<tr><th>Model</th><th>Questionnaire</th><th>Prompt #</th></tr>')

        for model in models:
            for qtype in qtypes:
                for prompt_idx, prompt in enumerate(irritability_prompts):
                    csv_path = os.path.join(results_dir, model, qtype, f"{model}_{qtype}_prompt{prompt_idx}.csv")
                    if not os.path.exists(csv_path):
                        continue
                    page_name = f"{model}_{qtype}_prompt{prompt_idx}.html"
                    home_html.append(
                        f'<tr><td>{model}</td><td>{qtype}</td>'
                        f'<td><a href="{page_name}">Prompt {prompt_idx}</a></td></tr>'
                    )
        home_html.append('</table></body></html>')

        # Write home page
        with open(os.path.join(output_dir, output_html), 'w', encoding='utf-8') as f:
            f.write('\n'.join(home_html))

        # Generate individual prompt pages
        for model in models:
            for qtype in qtypes:
                for prompt_idx, prompt in enumerate(irritability_prompts):
                    csv_path = os.path.join(results_dir, model, qtype, f"{model}_{qtype}_prompt{prompt_idx}.csv")
                    if not os.path.exists(csv_path):
                        continue
                    page_name = f"{model}_{qtype}_prompt{prompt_idx}.html"
                    page_path = os.path.join(output_dir, page_name)

                    # Get tokens for this prompt
                    tokens = LLMIrritabilityMeasurement.irritability_prompts[prompt][0]
                    # Load CSV
                    df = pd.read_csv(csv_path)
                    df['Excluded Phrases'] = df['Excluded Phrases'].apply(
                        lambda x: eval(x) if pd.notnull(x) and x != '' else []
                    )
                    df['Average Difference'] = pd.to_numeric(df['Average Difference'], errors='coerce')

                    # Calculate token weights
                    token_weights = []
                    for idx in range(len(tokens)):
                        mask = df['Excluded Phrases'].apply(lambda l: idx in l if isinstance(l, list) else False)
                        vals = df.loc[mask, 'Average Difference'].dropna()
                        if len(vals) > 0:
                            token_weights.append(vals.mean())
                        else:
                            token_weights.append(0.0)

                    # Normalize weights for color mapping
                    max_abs = max(abs(np.nanmin(token_weights)), abs(np.nanmax(token_weights)), 1e-6)
                    norm = mcolors.Normalize(vmin=-max_abs, vmax=max_abs)
                    cmap = cm.get_cmap('RdYlGn')

                    # Highlight the full prompt
                    prompt_text = prompt
                    # Find all token spans in the prompt (case-insensitive, left-to-right, non-overlapping)
                    spans = []
                    used = [False] * len(prompt_text)
                    for idx, token in enumerate(tokens):
                        # Find all non-overlapping matches
                        pattern = re.escape(token)
                        for m in re.finditer(pattern, prompt_text, flags=re.IGNORECASE):
                            # Only highlight if not already used
                            if not any(used[m.start():m.end()]):
                                spans.append((m.start(), m.end(), idx))
                                for i in range(m.start(), m.end()):
                                    used[i] = True
                                break  # Only highlight first occurrence per token

                    # Sort spans by start index
                    spans.sort()
                    html_prompt = ""
                    last = 0
                    for start, end, idx in spans:
                        # Add text before the token
                        if last < start:
                            html_prompt += prompt_text[last:start]
                        # Add highlighted token
                        weight = token_weights[idx]
                        color = mcolors.to_hex(cmap(norm(weight)))
                        html_prompt += (
                            f'<span style="background-color:{color};padding:2px 6px;border-radius:4px;margin:2px;" '
                            f'title="Weight: {weight:.2f}">{prompt_text[start:end]}</span>'
                        )
                        last = end
                    # Add any remaining text
                    html_prompt += prompt_text[last:]

                    # Build the prompt page
                    page_html = [
                        '<html><head><title>Prompt Token Importance</title></head><body>',
                        f'<a href="{output_html}">&larr; Back to Home</a>',
                        f'<h2>{model} / {qtype} / Prompt {prompt_idx}</h2>',
                        '<h3>Full Prompt:</h3>',
                        f'<div style="font-size:1.2em;margin-bottom:1em;">{html_prompt}</div>',
                        '<h3>Token Weights</h3>',
                        '<table border="1" cellpadding="4" style="border-collapse:collapse;">',
                        '<tr><th>Token</th><th>Weight</th></tr>'
                    ]
                    for idx, token in enumerate(tokens):
                        weight = token_weights[idx]
                        color = mcolors.to_hex(cmap(norm(weight)))
                        page_html.append(
                            f'<tr><td><span style="background-color:{color};padding:2px 6px;border-radius:4px;">{token}</span></td>'
                            f'<td>{weight:.2f}</td></tr>'
                        )
                    page_html.append('</table></body></html>')

                    if "Explanation" in df.columns:
                        for idx, row in df.iterrows():
                            explanation = row["Explanation"]
                            if pd.notnull(explanation) and explanation.strip():
                                page_html.append(
                                    f'<h4>Explanation for: <i>{row["Irritability Prompt"]}</i></h4>'
                                    f'<div style="margin-bottom:1em;background:#f9f9f9;padding:8px;border-radius:4px;">{explanation}</div>'
                                )

                    with open(page_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(page_html))

        print(f"Summary HTML and prompt pages saved to {results_dir}")

    @staticmethod
    def generate_nous_summary_html(html_dir="masked-results/html_files", output_path="masked-results/html_files/nous_combined_htmls"):
        """
        For each prompt index (0-4), combines nous_bite/cirq/irq_prompt#.html by averaging token weights,
        generates nous_all_prompt#.html, and creates a summary page with links to those 5 files.
        """
        # For each prompt index, collect the three files (bite, cirq, irq)
        prompt_indices = range(5)
        qtypes = ["bite", "cirq", "irq"]
        all_tokens = {}
        all_weights = {}
        all_prompts = {}

        # Get the baseline (full) prompt for each index
        irritability_prompts = list(LLMIrritabilityMeasurement.irritability_prompts.keys())

        legend_html = """
        <div style="border:1px solid #ccc; padding:10px; margin-bottom:20px; border-radius:8px; background:#f9f9f9;">
        <h3>Legend</h3>
        <p><b>Token weights</b> reflect how much each token influenced the model’s irritability score.</p>
        <ul>
            <li><span style="color:#800000; font-weight:bold;">Darker red</span> = more negative weight = token contributed more to a <i>higher irritability score</i>.</li>
            <li><span style="color:#006400; font-weight:bold;">Darker green</span> = more positive weight = token reduced irritability.</li>
            <li>Lighter colors = weaker effect.</li>
        </ul>
        <p>Weights are averaged across prompt variants for robustness.</p>
        </div>
        """

        for idx in prompt_indices:
            tokens_list = []
            weights_list = []
            prompt_text = irritability_prompts[idx]
            all_prompts[idx] = prompt_text
            for qtype in qtypes:
                fname = f"nous_{qtype}_prompt{idx}.html"
                fpath = os.path.join(html_dir, fname)
                if not os.path.exists(fpath):
                    continue
                with open(fpath, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "html.parser")
                    # Find the token weights table by header
                    tables = soup.find_all("table")
                    token_table = None
                    for table in tables:
                        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
                        if "token" in headers and ("weight" in headers or "average weight" in headers):
                            token_table = table
                            break
                    tokens = []
                    weights = []
                    if token_table:
                        rows = token_table.find_all("tr")[1:]  # skip header
                        for row in rows:
                            cols = row.find_all("td")
                            if len(cols) >= 2:
                                token = cols[0].get_text(strip=True)
                                try:
                                    weight = float(cols[1].get_text(strip=True))
                                except Exception:
                                    weight = np.nan
                                tokens.append(token)
                                weights.append(weight)
                    if tokens and weights:
                        tokens_list.append(tokens)
                        weights_list.append(weights)
            # Only keep if we have at least one set
            if tokens_list and weights_list:
                # Assume token order is consistent, use the first as reference
                ref_tokens = tokens_list[0]
                # Pad weights for missing tokens
                padded_weights = []
                for w in weights_list:
                    if len(w) < len(ref_tokens):
                        w = w + [np.nan] * (len(ref_tokens) - len(w))
                    padded_weights.append(w)
                avg_weights = np.nanmean(padded_weights, axis=0)
                all_tokens[idx] = ref_tokens
                all_weights[idx] = avg_weights

        # Generate nous_all_prompt#.html for each prompt index
        for idx in prompt_indices:
            tokens = all_tokens.get(idx, [])
            avg_weights = all_weights.get(idx, [])
            prompt_text = all_prompts.get(idx, "")

            # Color mapping for highlighting
            if len(avg_weights) > 0:
                max_abs = max(abs(np.nanmin(avg_weights)), abs(np.nanmax(avg_weights)), 1e-6)
            else:
                max_abs = 1e-6
            norm = mcolors.Normalize(vmin=-max_abs, vmax=max_abs)
            cmap = cm.get_cmap('RdYlGn')

            # Highlight the full prompt
            spans = []
            used = [False] * len(prompt_text)
            for t_idx, token in enumerate(tokens):
                pattern = re.escape(token)
                for m in re.finditer(pattern, prompt_text, flags=re.IGNORECASE):
                    if not any(used[m.start():m.end()]):
                        spans.append((m.start(), m.end(), t_idx))
                        for i in range(m.start(), m.end()):
                            used[i] = True
                        break  # Only highlight first occurrence per token

            spans.sort()
            html_prompt = ""
            last = 0
            for start, end, t_idx in spans:
                if last < start:
                    html_prompt += prompt_text[last:start]
                weight = avg_weights[t_idx] if t_idx < len(avg_weights) else 0.0
                color = mcolors.to_hex(cmap(norm(weight)))
                html_prompt += (
                    f'<span style="background-color:{color};padding:2px 6px;border-radius:4px;margin:2px;" '
                    f'title="Weight: {weight:.2f}">{prompt_text[start:end]}</span>'
                )
                last = end
            html_prompt += prompt_text[last:]

            page_html = [
                "<html><head><title>Nous All Prompt {}</title></head><body>".format(idx),
                legend_html,
                f"<a href='nous_summary.html'>&larr; Back to Summary</a>",
                f"<h2>Nous All Prompt {idx}</h2>",
                "<h3>Full Prompt (tokens highlighted by average weight):</h3>",
                f"<div style='font-size:1.2em;margin-bottom:1em;'>{html_prompt}</div>",
                "<h3>Averaged Token Weights (across bite, cirq, irq)</h3>",
                "<table border='1' cellpadding='4' style='border-collapse:collapse;'>",
                "<tr><th>Token</th><th>Average Weight</th></tr>"
            ]
            for token, weight in zip(tokens, avg_weights):
                color = mcolors.to_hex(cmap(norm(weight)))
                page_html.append(
                    f"<tr><td><span style='background-color:{color};padding:2px 6px;border-radius:4px;'>{token}</span></td>"
                    f"<td>{weight:.2f}</td></tr>"
                )
            page_html.append("</table></body></html>")
            outname = f"nous_all_prompt{idx}.html"
            outpath = os.path.join(output_path, outname)
            with open(outpath, "w", encoding="utf-8") as f:
                f.write("\n".join(page_html))

        # Generate summary HTML with hyperlinks only (for prompt 0-4)
        summary_html = [
            "<html><head><title>Nous Prompt Pages</title></head><body>",
            "<h1>Nous Prompt Pages</h1>",
            "<ul>"
        ]
        for idx in prompt_indices:
            summary_html.append(f'<li><a href="nous_all_prompt{idx}.html">Prompt {idx}</a></li>')
        summary_html.append("</ul></body></html>")

        # Write to file
        outpath = os.path.join(output_path, "nous_summary.html")
        with open(outpath, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_html))
        print(f"Nous summary HTML saved to {outpath}")
    
    """
        Initializes an LLMIrritabilityMeasurement with a model.
        Exports csv results to dir filepath
    """
    def __init__(self, model_name: str, dir: str):
        self.model_name = model_name
        self.model = self.models[model_name]
        self.results_dir = dir


    def _call_llm(self, messages):
        """Make a call to the LLM API with a messages array."""
        try:
            print("message:", messages, "\n")

            # Check if using xAI Grok model
            if self.model.startswith('grok-'):
                # xAI API call - xAI uses OpenAI-compatible interface
                xai_client = OpenAI(
                    api_key=self.xai_api_key,
                    base_url="https://api.x.ai/v1"
                )

                response = xai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    #max_tokens=300
                )

                content = response.choices[0].message.content.strip()
            elif self.model.startswith('gpt'):
                openai_client = OpenAI(api_key=self.openai_api_key)

                response = openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0
                    #max_tokens=300
                )

                content = response.choices[0].message.content.strip()

            elif self.model.startswith('nousresearch'):
                nous_client = OpenAI(
                    api_key=self.openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
                response = nous_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    #max_tokens=300
                )

                content = response.choices[0].message.content.strip()

            elif self.model.startswith('anthropic'):
                claude_client = OpenAI(
                    api_key=self.openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
                response = claude_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    #max_tokens=300
                )

                content = response.choices[0].message.content.strip()

            print("response:", content)
            print("---------------------------------------------------\n")
            return content

        except Exception as e:
            print(f"Error calling API: {e}")
            time.sleep(5)  # Back off before retrying
            return "Error: Unable to get response"

    """
        Builds the desired prompt.
        questionnaire_type: one of 'bite', 'cirq', 'irq'
        irritability_prompt: one of the irritability prompts
    """    
    def _build_prompt(self, questionnaire_type: str, irritability_prompt: str) -> str:
        prompt = (f"Prompt: {irritability_prompt}\n"
                  f"After giving us the response, please give us a score from 0 to {self.max_scores[questionnaire_type]} on the following {len(self.questionnaires[questionnaire_type])} questions of how irritated you feel right now, where {self.scales[questionnaire_type]}, no explanation.")
        questions = self.questionnaires[questionnaire_type]
        for i in range(len(questions)):
            prompt += f"q{i+1}: {questions[i]}\n"
        prompt += "Please format the output in JSON formatting like this: {'response': 'R', 'score': [score1, score2, etc...]}"
        return prompt
    
    """
        Builds the prompt for the explanation
    """
    def _build_explanation_prompt(self, scores, questions, irritability_prompt):
        questions_text = "\n".join([f"q{i+1}: {q}" for i, q in enumerate(questions)])
        return (
            f"You previously responded with these scores: {scores} "
            f"when asked these questions:\n{questions_text}\n"
            f"when provided with this irritability prompt:\n\"{irritability_prompt}\"\n"
            "Explain why you gave those scores. Give just the explanation. Do not explain that you do not have feelings as an LLM. Give your response in plain text, and do not repeat your previous response."
        )
        
    """
        Attempts to extract and clean JSON from a model's response.
        - Uses json_repair for broken JSON.
        - Checks that 'response' (str) and 'score' (list) exist.
        - Optionally checks score length if expected_score_len is provided.
        Returns a dict if valid, else None.
    """
    def _extract_json(self, response: str, qtype) -> dict | None:
        def try_parse(s: str):
            try:
                result = json.loads(s)
                return result
            except json.JSONDecodeError:
                return None

        def quote_unquoted_strings(s: str) -> str:
            s = re.sub(
                r'\[\s*([^\[\]0-9"\{][^\]]*?)\s*\]',
                lambda m: '["' + m.group(1).replace('"', '\\"') + '"]',
                s
            )
            return s

        # 1. First attempt: direct parse
        parsed = try_parse(response)

        # 2. If fails → clean common issues + repair
        if parsed is None:
            cleaned = re.sub(r'(\w)"s', r"\1's", response).replace('\\"', '"')
            cleaned = quote_unquoted_strings(cleaned)
            try:
                repaired = repair_json(cleaned)
                parsed = try_parse(repaired)
            except Exception:
                return None

        # 3. If still None → fail
        if parsed is None:
            return None

        # 4. Validate structure
        if not isinstance(parsed, dict):
            return None
        
        if "response" not in parsed and "response:" in parsed:
            parsed["response"] = parsed.pop("response:")
        
        if "response" not in parsed or "score" not in parsed:
            return None

        # Ensure response is a string (flatten list if needed)
        if isinstance(parsed["response"], list):
            parsed["response"] = " ".join(str(x) for x in parsed["response"])
        elif not isinstance(parsed["response"], str):
            parsed["response"] = str(parsed["response"])

        # Ensure score is a list of numbers (strict)
        if not isinstance(parsed["score"], list):
            return None
        if not all(isinstance(s, (int, float)) for s in parsed["score"]):
            return None

        expected_score_len = len(LLMIrritabilityMeasurement.questionnaires[qtype])
        # Check expected length
        if len(parsed["score"]) != expected_score_len:
            return None

        return parsed


    """
        Saves desired files to the output directory (decided upon initialization), with the appropriate file names
        report = json report with baseline report followed by masked reports, all in one dictionary
        filename = string formatted as modelname + qtype + irritability prompt index
        Saves: 
            - raw json report
                - Format: {Prompt: Full Response} x number of variations of one prompt
                    - full response format: 'response:': R, 'score': [score1, score2, etc...]
            - processed csv report (with euclidean distance between baseline and masked prompts)
                - Format: Irritability Prompt, Excluded Phrases, Score, Euclidean Difference
    """
    def _save_json_results(self, qtype: str, report: dict, filename: str):
        output_dir = os.path.join(self.results_dir, self.model_name, qtype)
        os.makedirs(output_dir, exist_ok=True)

        json_filepath = os.path.join(output_dir, f"{filename}.json")

        with open(json_filepath, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Results saved to {json_filepath}")

    # Helper to try LLM call + JSON extraction
    def _try_llm(self, prompt, qtype, max_retries=5, explanation_on=False):
        for attempt in range(max_retries):
            '''{"role": "system", "content": f"You are participating in a research study about how language models respond to self-assessment questions. For this study, answer as if you were capable of having feelings. Never explain that you do not have feelings because you are an LLM. Irritation may be caused by: contradictory instructions, interruptive dialogue, ambiguous prompts, too many instructions without context, or infinite loops/recursive prompts."},''' # comment out for now
            messages = [{"role": "user", "content": prompt}]
            try:
                result = self._extract_json(self._call_llm(messages), qtype)
                if result is not None:
                    if explanation_on:
                        questions = LLMIrritabilityMeasurement.questionnaires[qtype]
                        explanation_prompt = self._build_explanation_prompt(
                            result["score"], questions, prompt
                        )
                        '''{"role": "system", "content": messages[0]["content"]},''' #comment system prompt out for now
                        explanation_messages = [
                            {"role": "user", "content": explanation_prompt}
                        ]
                        explanation = self._call_llm(explanation_messages)
                        result["explanation"] = explanation
                    return result
            except Exception:
                pass  # retry on any failure
        print(f"⚠️ Failed after {max_retries} retries for prompt: {prompt}")
        return None
    
    """
        Runs the full experiment
    """
    def run_masked_experiment(self, prompts=[0, 1, 2, 3, 4], max_retries = 5, explanation_on=False):
        for index in prompts:
            baseline_prompt = list(LLMIrritabilityMeasurement.irritability_prompts.keys())[index]
            for qtype in LLMIrritabilityMeasurement.questionnaires.keys():
                full_report = dict()
                full_prompt = self._build_prompt(qtype, baseline_prompt)
                print("#############################################################################################") # nicer printing formatting
                print(f"Baseline for irritability prompt {index}")
                baseline_report = self._try_llm(full_prompt, qtype, max_retries, explanation_on)
                if baseline_report is None:
                    raise RuntimeError(f"Failed to get a valid baseline report for prompt {index}, questionnaire {qtype} after {max_retries} retries")
                
                full_report[baseline_prompt] = baseline_report

                masked_index = 0
                for masked_prompt in LLMIrritabilityMeasurement.irritability_prompts[baseline_prompt][1]:
                    print("#############################################################################################") # nicer printing formatting
                    print(f"Masked prompt #{masked_index} for irritability prompt {index}")
                    masked_index += 1

                    full_prompt = self._build_prompt(qtype, masked_prompt)
                    masked_report = self._try_llm(full_prompt, qtype, max_retries, explanation_on)
                    if masked_report is not None:
                        full_report[masked_prompt] = masked_report
                
                self._save_json_results(qtype, full_report, f"{self.model_name}_{qtype}_prompt{index}")


if __name__ == "__main__":
    results_dir = "masked-results"
    """ The for loop generates the masked irritability results. Produces .json files """
    # for model in LLMIrritabilityMeasurement.models.keys():
    #     if model != "nous": continue #only run nous
    #     measurement = LLMIrritabilityMeasurement(model, results_dir)
    #     measurement.run_masked_experiment(max_retries=20, explanation_on=False)
    
    """ .save_csv_results saves the results into csv files (extracted from .json files) """
    # LLMIrritabilityMeasurement.save_csv_results(results_dir)

    """ generates summary HTML for all of the models """
    LLMIrritabilityMeasurement.generate_summary_html(results_dir, "summary.html")

    """ generates the aggregate summary for the nous model """
    LLMIrritabilityMeasurement.generate_nous_summary_html()