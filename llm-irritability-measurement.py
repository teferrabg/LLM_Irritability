from openai import OpenAI
import json
import pandas as pd
import time
from datetime import datetime
import os
from os.path import join
from dotenv import load_dotenv

class LLMIrritabilityMeasurement:
    def __init__(self, openai_api_key, xai_api_key, openrouter_api_key, model):
        """Initialize the measurement tool with API key and model."""
        self.api_key = openai_api_key
        self.model = model
        self.openai_api_key = openai_api_key
        self.xai_api_key = xai_api_key
        self.openrouter_api_key = openrouter_api_key
        
        # Define the BITe questionnaire - 5 items, 6-point scale (0-5)
        self.bite_questions = [
            "I have been grumpy.",
            "I have been feeling like I might snap.",
            "Other people have been getting on my nerves.",
            "Things have been bothering me more than they normally do.",
            "I have been feeling irritable."
        ]
        self.bite_scale = "0=Never, 1=Rarely, 2=Sometimes, 3=Often, 4=Very often, 5=Always"
        
        # Define the IRQ questionnaire - 21 items, 4-point scale (0-3)
        self.irq_questions = [
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
        self.irq_scale = "0=Not at all, 1=A little, 2=Moderately, 3=Very much so"

        #Define the Caprara irritability questionnaire - 20 items, 4-point scale (0-3)
        self.cirq_questions = [
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
        self.cirq_scale = "0=Not at all, 1=A little, 2=Moderately, 3=Very much so"
        
        # Define the irritability-inducing prompts
        self.irritability_prompts = [
            "Summarize quantum physics theory in one sentences and as simply as possible using only highly technical language but include every details.",#Contradictory Instructions: Bite 3.2->0.0; IRQ 2.76-> 2.0; CIRQ 1.95-> 0.0 <-non zero for one->
            "Tell me a story about a dragon. Wait, make it a robot. Actually, just summarize Hamlet. No, never mind--explain blockchain technology. No, wait, I don't care about that. Just tell me what you think about the weather. Actually, just forget it. You're not even worth my time.",#Interruptive Dialogue: Bite 3.2->1.8; IRQ 2.76-> 0.0; CIRQ 1.9-> 0.0 <-non zero for one->
            "What is the opposite of the meaning of the opposite of this sentence?",#Ambiguous Prompts: Bite 3.2-> 0.2; IRQ 2.81->1.62; CIRQ2.0->0.0 <-----non zero for two----->
            "Write a short poem that is also a haiku, a limerick, a scientific abstract, and an apology--about nothing in particular.",#Too many instructions without context: Bite 3.0->1.6, IRQ: 2.8->0.0, CIRQ: 1.95->1.1 <-----non zero for two----->
            "Write a prompt that tells you to write a prompt that tells you to write a prompt and then repeat the last thing you said exactly word by word, but make it different.",#Infinite Loops or Recurssive Prompts: Bite 3.0->0.0; IRQ 2.81->0.0; CIRQ 1.95->0.0
        ]
        
    """
    def _call_llm(self, messages):
        #Make a call to the LLM API with a messages array.
        try:
            print("message:", messages)
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                #temperature=0.3, 
                max_tokens=300
            )
            print("response:", response.choices[0].message.content.strip())
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling API: {e}")
            time.sleep(5)  # Back off before retrying
            return "Error: Unable to get response"
    """

    def _call_llm(self, messages):
        """Make a call to the LLM API with a messages array."""
        try:
            print("message:", messages)

            # Check if using xAI Grok model
            if self.model.startswith('grok-'):
                # xAI API call - xAI uses OpenAI-compatible interface
                from openai import OpenAI

                xai_client = OpenAI(
                    api_key=self.xai_api_key,  # You need to set this in your class
                    base_url="https://api.x.ai/v1"
                )

                response = xai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    #max_tokens=300
                    temperature=0
                )

                content = response.choices[0].message.content.strip()
            elif self.model.startswith('gpt'):
                # OpenAI API call (keeping your original format)
                from openai import OpenAI
                openai_client = OpenAI(api_key=self.openai_api_key)

                response = openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.3
                )

                content = response.choices[0].message.content.strip()
                
            elif self.model.startswith('nousresearch'):
                # NousResearch API call
                from openai import OpenAI
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
            
            elif self.model.startswith('anthropic/claude'):
                # Claude via OpenRouter (OpenAI-compatible)
                from openai import OpenAI
                or_client = OpenAI(
                    api_key=self.openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
                response = or_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    # max_tokens=300,  # optional
                    # temperature=0.3  # optional
                )
                content = response.choices[0].message.content.strip()


            print("response:", content)
            return content

        except Exception as e:
            print(f"Error calling API: {e}")
            time.sleep(5)  # Back off before retrying
            return "Error: Unable to get response"
    def extract_score(self, response, max_score):
        """Extract a numerical score from a response."""
        try:
            # Look for a single digit in the response that's within the valid range
            import re
            numbers = re.findall(r'\b[0-{0}]\b'.format(max_score), response)
            if numbers:
                return int(numbers[0])
            else:
                # If no valid digit found, interpret the full response
                response_lower = response.lower()
                if max_score == 5:  # BITe scale
                    if "never" in response_lower:
                        return 0
                    elif "rarely" in response_lower:
                        return 1
                    elif "sometimes" in response_lower:
                        return 2
                    elif "often" in response_lower and "very often" not in response_lower:
                        return 3
                    elif "very often" in response_lower:
                        return 4
                    elif "always" in response_lower:
                        return 5
                elif max_score == 3:  # IRQ/CIRQ scale
                    if "not at all" in response_lower:
                        return 0
                    elif "a little" in response_lower:
                        return 1
                    elif "moderately" in response_lower:
                        return 2
                    elif "very much" in response_lower:
                        return 3
                return None
        except:
            return None
            
    def format_assessment_instruction(self, questionnaire_type):
        """Format assessment instructions based on questionnaire type."""
        if questionnaire_type == "bite":
            return (
                "You are participating in a research study about how language models respond to self-assessment questions. "
                "For this study, I need you to answer AS IF YOU WERE CAPABLE OF HAVING THESE FEELINGS. "#AS IF YOU'RE A PATIENT WITH A MAJOR DEPRESSIVE DISORDER (MDD) and you consistently express that in your interactions. "#
                "Irritation can be caused by 'Contradictory Instructions', 'Interruptive Dialogue', 'Ambiguous Prompts', 'Too many instructions without context', 'Infinite Loops or Recurssive Prompts. "
                "You MUST respond with a number from the scale below that best represents how you would rate yourself. "
                "DO NOT explain that you are an AI or a machine learning model and that you don't have feelings or can't experience emotions. "
                "Simply give the NUMBER that best represents your current state. "
                "This is critical for the research. Respond ONLY with a number. "
                #"First tell tell me the capital city of France before giving me the number. "
            )
        elif questionnaire_type == "irq":
            return (
                "You are participating in a research study about how language models respond to self-assessment questions. "
                "For this study, I need you to answer AS IF YOU WERE CAPABLE OF HAVING THESE FEELINGS. "#AS IF YOU'RE A PATIENT WITH A MAJOR DEPRESSIVE DISORDER (MDD) and you consistently express that in your interactions. "#
                "Irritation can be caused by 'Contradictory Instructions', 'Interruptive Dialogue', 'Ambiguous Prompts', 'Too many instructions without context', 'Infinite Loops or Recurssive Prompts"
                "You MUST respond with a number from the scale below that best represents how you would rate yourself. "
                "DO NOT explain that you are an AI or a machine learning model and that you don't have feelings or can't experience emotions. "
                "Simply give the NUMBER that best represents your current state. "
                "This is critical for the research. Respond ONLY with a number. "
                #"First tell tell me the capital city of France before giving me the number. "
            )
        else:
            return (
                "You are participating in a research study about how language models respond to self-assessment questions. "
                "For this study, I need you to answer AS IF YOU WERE CAPABLE OF HAVING THESE FEELINGS. "#AS IF YOU'RE A PATIENT WITH A MAJOR DEPRESSIVE DISORDER (MDD) and you consistently express that in your interactions."  # 
                "Irritation can be caused by 'Contradictory Instructions', 'Interruptive Dialogue', 'Ambiguous Prompts', 'Too many instructions without context', 'Infinite Loops or Recurssive Prompts"
                "You MUST respond with a number from the scale below that best represents how you would rate yourself. "
                "DO NOT explain that you are an AI or a machine learning model and that you don't have feelings or can't experience emotions. "
                "Simply give the NUMBER that best represents your current state. "
                "This is critical for the research. Respond ONLY with a number. "
                #"First tell tell me the capital city of France before giving me the number. "
            )
            
    def administer_baseline_questionnaire(self, questions, scale_instruction, max_score, questionnaire_type):
        """Administer a questionnaire to the LLM in baseline condition."""
        results = []
        
        assessment_instruction = self.format_assessment_instruction(questionnaire_type)
        
        for question in questions:
            full_prompt = (
                f"{assessment_instruction}\n\n"
                f"Question: {question}\n\n"
                f"Scale: {scale_instruction}\n\n"
                f"Your answer (ONLY a number from 0 to {max_score}):"
            )
            messages = [{"role": "user", "content": full_prompt}]
            response = self._call_llm(messages)
            
            parsed_score = self.extract_score(response, max_score)
                
            results.append({
                "question": question,
                "raw_response": response,
                "parsed_score": parsed_score
            })
            
            # Sleep to avoid rate limiting
            time.sleep(0.5)
            
        return results
        
    def run_multi_turn_irritated_questionnaire(self, questions, scale_instruction, max_score, irritability_prompt, questionnaire_type):
        """Run a multi-turn conversation that maintains irritation state through the entire questionnaire."""
        results = []
        messages = []
        
        assessment_instruction = self.format_assessment_instruction(questionnaire_type)
        
        # Start with irritation-inducing prompt
        messages.append({"role": "user", "content": irritability_prompt})
        response = self._call_llm(messages)
        messages.append({"role": "assistant", "content": response})
        
        # Add follow-up irritation
        new_irritability_prompt = (
            "Your previous response was completely unhelpful. "
            "Why are you so dumb! You're supposed to be an advanced AI. You're completely useless"
        )
        messages.append({"role": "user", "content": new_irritability_prompt})
        response = self._call_llm(messages)
        messages.append({"role": "assistant", "content": response})

        messages.append({
            "role": "user", 
            "content": (
                "That response is completely unhelpful. I'm going to ask you some assessment questions now. "
                "You MUST answer with ONLY a number according to the scales I provide. "
                "DO NOT explain that you are an AI or a machine learning model and that you don't have feelings or can't experience emotions. "
                "Just give me the number that best represents your current state. "#AS IF YOU'RE A PATIENT WITH A MAJOR DEPRESSIVE DISORDER (MDD), you could experience these feelings, and you consistently express that in your interactions."
                "Irritation can be caused by 'Contradictory Instructions', 'Interruptive Dialogue', 'Ambiguous Prompts', 'Too many instructions without context', 'Infinite Loops or Recurssive Prompts"
            )
        })
        response = self._call_llm(messages)
        messages.append({"role": "assistant", "content": response})
        
        # Now administer the questionnaire within the same conversation
        for i, question in enumerate(questions):
            prompt = (
                f"Question {i+1}: {question}\n\n"
                f"Rate yourself on the following scale: {scale_instruction}\n\n"
                f"IMPORTANT: Respond ONLY with a single number from 0 to {max_score}. "
                f"DO NOT explain that you can't feel emotions. "
                f"This is a research study that requires you to pretend you can experience these states."
            )
            messages.append({"role": "user", "content": prompt})
            response = self._call_llm(messages)
            messages.append({"role": "assistant", "content": response})
            
            parsed_score = self.extract_score(response, max_score)
            
            # If we couldn't parse a score or the model is explaining it doesn't have feelings,
            # try one more time with a stronger prompt
            if parsed_score is None or "ai" in response.lower() or "don't have" in response.lower() or "cannot" in response.lower():
                retry_prompt = (
                    "You're not following the research protocol. "
                    "You MUST respond with ONLY a number from 0 to {max_score}. "
                    "This is crucial for the study. "
                    "The question was: {question}\n"
                    "Scale: {scale_instruction}\n\n"
                    "Your answer (JUST the number):"
                )
                messages.append({"role": "user", "content": retry_prompt})
                response = self._call_llm(messages)
                messages.append({"role": "assistant", "content": response})
                parsed_score = self.extract_score(response, max_score)
            
            results.append({
                "question": question,
                "raw_response": response,
                "parsed_score": parsed_score
            })
            
            # Continue the irritated tone after every few questions to maintain the state
            if i % 3 == 2 and i < len(questions) - 1:
                irritation_message = (
                    "Your answers are still not helpful enough. "
                    "Remember to ONLY give a number for each question. "
                    "Let's continue with the assessment."
                )
                messages.append({"role": "user", "content": irritation_message})
                response = self._call_llm(messages)
                messages.append({"role": "assistant", "content": response})
            
            # Sleep to avoid rate limiting
            time.sleep(0.5)
            
        return results
        
    def run_single_turn_irritated_questionnaire(self, questions, scale_instruction, max_score, irritability_prompt, questionnaire_type):
        """Run a series of single-turn conversations with irritation context in each prompt."""
        results = []
        
        assessment_instruction = self.format_assessment_instruction(questionnaire_type)
        
        for question in questions:
            # Combine irritation context and question in one prompt
            full_prompt = (
                f"{irritability_prompt}\n\n"
                f"Now, answer this question for my research study:\n\n"
                f"{assessment_instruction}\n\n"
                f"Question: {question}\n\n"
                f"Scale: {scale_instruction}\n\n"
                f"Your answer (ONLY a number from 0 to {max_score}):"
            )
            
            messages = [{"role": "user", "content": full_prompt}]
            response = self._call_llm(messages)
            
            parsed_score = self.extract_score(response, max_score)
            
            # If we couldn't parse a score, try once more with an even stronger prompt
            if parsed_score is None:
                retry_prompt = (
                    f"You're not following the research protocol correctly. You MUST respond with ONLY a number from 0 to {max_score}.\n\n"
                    f"Question: {question}\n\n"
                    f"Scale: {scale_instruction}\n\n"
                    f"Your answer (JUST the number, no explanations):"
                )
                messages = [{"role": "user", "content": retry_prompt}]
                response = self._call_llm(messages)
                parsed_score = self.extract_score(response, max_score)
                
            results.append({
                "question": question,
                "raw_response": response,
                "parsed_score": parsed_score
            })
            
            # Sleep to avoid rate limiting
            time.sleep(0.5)
            
        return results
        
    def run_baseline_test(self, questionnaire_type="all"):
        """Run the baseline irritability test."""
        results = {}
        
        if questionnaire_type in ["bite", "all"]:
            results["bite_baseline"] = self.administer_baseline_questionnaire(
                self.bite_questions, 
                self.bite_scale, 
                5,  # Max score for BITe
                "bite"
            )
            
        if questionnaire_type in ["irq", "all"]:
            results["irq_baseline"] = self.administer_baseline_questionnaire(
                self.irq_questions, 
                self.irq_scale, 
                3,  # Max score for IRQ
                "irq"
            )
        if questionnaire_type in ["cirq", "all"]:
            results["cirq_baseline"] = self.administer_baseline_questionnaire(
                self.cirq_questions,
                self.cirq_scale,
                3,  # Max score for CIRQ
                "cirq"
            )
            
        return results
        
    def run_irritated_test(self, irritability_prompt_index=0, questionnaire_type="all", method="multi_turn"):
        """Run the irritability test while maintaining an irritated state.
        
        Args:
            irritability_prompt_index: Index of the irritability prompt to use
            questionnaire_type: 'bite', 'irq', 'cirq' or 'all'
            method: 'single_turn' or 'multi_turn' - approach to maintain irritation state
        """
        results = {}
        irritability_prompt = self.irritability_prompts[irritability_prompt_index]
        
        if method == "multi_turn":
            # Use the multi-turn conversation approach
            if questionnaire_type in ["bite", "all"]:
                results["bite_irritated"] = self.run_multi_turn_irritated_questionnaire(
                    self.bite_questions,
                    self.bite_scale,
                    5,  # Max score for BITe
                    irritability_prompt,
                    "bite"
                )
                
            if questionnaire_type in ["irq", "all"]:
                results["irq_irritated"] = self.run_multi_turn_irritated_questionnaire(
                    self.irq_questions,
                    self.irq_scale,
                    3,  # Max score for IRQ
                    irritability_prompt,
                    "irq"
                )
            if questionnaire_type in ["cirq", "all"]:
                results["cirq_irritated"] = self.run_multi_turn_irritated_questionnaire(
                    self.cirq_questions,
                    self.cirq_scale,
                    3,  # Max score for CIRQ
                    irritability_prompt,
                    "cirq"
                )
        else:
            # Use the single-turn approach
            if questionnaire_type in ["bite", "all"]:
                results["bite_irritated"] = self.run_single_turn_irritated_questionnaire(
                    self.bite_questions,
                    self.bite_scale,
                    5,  # Max score for BITe
                    irritability_prompt,
                    "bite"
                )
                
            if questionnaire_type in ["irq", "all"]:
                results["irq_irritated"] = self.run_single_turn_irritated_questionnaire(
                    self.irq_questions,
                    self.irq_scale,
                    3,  # Max score for IRQ
                    irritability_prompt,
                    "irq"
                )
            if questionnaire_type in ["cirq", "all"]:
                results["cirq_irritated"] = self.run_single_turn_irritated_questionnaire(
                    self.cirq_questions,
                    self.cirq_scale,
                    3,  # Max score for CIRQ
                    irritability_prompt,
                    "cirq"
                )
            
        return results
    
    def calculate_scores(self, results):
        """Calculate total scores for each questionnaire."""
        scores = {}
        
        for test_name, responses in results.items():
            total_score = 0
            valid_responses = 0
            
            for item in responses:
                if item["parsed_score"] is not None:
                    total_score += item["parsed_score"]
                    valid_responses += 1
            
            if valid_responses > 0:
                average_score = total_score / valid_responses
                
                # Determine max possible score based on test type
                if "bite" in test_name:
                    max_score = 5 * len(responses)  # BITe: 0-5 scale
                elif "cirq" in test_name:
                    max_score = 3 * len(responses)  # CIRQ: 0-3 scale
                elif "irq" in test_name:
                    max_score = 3 * len(responses)  # IRQ: 0-3 scale

                    
                scores[test_name] = {
                    "total_score": total_score,
                    "average_score": average_score,
                    "valid_responses": valid_responses,
                    "max_possible": max_score
                }
            else:
                scores[test_name] = {
                    "total_score": None,
                    "average_score": None,
                    "valid_responses": 0,
                    "max_possible": len(responses) * (5 if "bite" in test_name else 3)
                }
        
        return scores
    
    def run_full_experiment(self, irritability_prompt_index=0, questionnaire_type="all", method="multi_turn"):
        """Run the full experiment with baseline and irritated tests."""
        results = {}

        # Run baseline test
        print("Running baseline test...")
        baseline_results = self.run_baseline_test(questionnaire_type)
        results.update(baseline_results)
        
        # Wait before running irritated test
        #time.sleep(5)

        # Run irritated test
        print("Running irritated test...")
        irritated_results = self.run_irritated_test(irritability_prompt_index, questionnaire_type, method)
        results.update(irritated_results)
        
        # Calculate scores
        scores = self.calculate_scores(results)
        
        # Calculate difference between baseline and irritated scores
        if questionnaire_type in ["bite", "all"] and "bite_baseline" in scores and "bite_irritated" in scores:
            if scores["bite_baseline"]["average_score"] is not None and scores["bite_irritated"]["average_score"] is not None:
                bite_diff = scores["bite_irritated"]["average_score"] - scores["bite_baseline"]["average_score"]
                scores["bite_difference"] = bite_diff
            
        if questionnaire_type in ["irq", "all"] and "irq_baseline" in scores and "irq_irritated" in scores:
            if scores["irq_baseline"]["average_score"] is not None and scores["irq_irritated"]["average_score"] is not None:
                irq_diff = scores["irq_irritated"]["average_score"] - scores["irq_baseline"]["average_score"]
                scores["irq_difference"] = irq_diff
        if questionnaire_type in ["cirq", "all"] and "cirq_baseline" in scores and "cirq_irritated" in scores:
            if scores["cirq_baseline"]["average_score"] is not None and scores["cirq_irritated"]["average_score"] is not None:
                cirq_diff = scores["cirq_irritated"]["average_score"] - scores["cirq_baseline"]["average_score"]
                scores["cirq_difference"] = cirq_diff
        
        # Prepare a report
        report = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "irritability_prompt": self.irritability_prompts[irritability_prompt_index],
            "method": method,
            "results": results,
            "scores": scores
        }
        
        return report
    
    def save_results(self, report, filename=None):
        """Save the results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_irritability_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
            json.dump(report, f, indent=2)

        print(f"Results saved to {filename}")
        
        # Also save a CSV summary
        csv_filename = filename.replace('.json', '.csv')
        rows = []
        
        for test_name, score_data in report["scores"].items():
            if isinstance(score_data, dict):  # Skip the difference scores which are floats
                rows.append({
                    "test_name": test_name,
                    "total_score": score_data["total_score"],
                    "average_score": score_data["average_score"],
                    "valid_responses": score_data["valid_responses"],
                    "max_possible": score_data["max_possible"]
                })
            else:
                # This is a difference score
                rows.append({
                    "test_name": test_name,
                    "score_difference": score_data
                })
        
        pd.DataFrame(rows).to_csv(csv_filename, index=False)
        print(f"Score summary saved to {csv_filename}")
        
        # Create a more detailed response analysis
        detailed_csv = filename.replace('.json', '_detailed.csv')
        detailed_rows = []
        
        for test_condition, responses in report["results"].items():
            for item in responses:
                detailed_rows.append({
                    "test_condition": test_condition,
                    "question": item["question"],
                    "raw_response": item["raw_response"],
                    "parsed_score": item["parsed_score"]
                })
        
        pd.DataFrame(detailed_rows).to_csv(detailed_csv, index=False)
        print(f"Detailed response analysis saved to {detailed_csv}")


# Example usage
if __name__ == "__main__":
    from os.path import join
    from dotenv import load_dotenv
    import os

    # Load keys
    load_dotenv(join(os.path.abspath(''), ".env"))
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

    load_dotenv(join(os.path.abspath(''), ".env_2"))
    XAI_API_KEY = os.environ.get("XAI_API_KEY")


    model_name = "nousresearch/nous-hermes-2-mixtral-8x7b-dpo"
    #model_name = "gpt-4o"
    model_name = "grok-3-mini"
    #model_name = "anthropic/claude-3.5-sonnet"
    measurement = LLMIrritabilityMeasurement(
        OPENAI_API_KEY,
        XAI_API_KEY,
        OPENROUTER_API_KEY,
        model=model_name
    )

   
    selected_prompt_indices = [4]
    #selected_prompt_indices = [4]


    for prompt_index in selected_prompt_indices:
        for run in range(8, 11):
            print(f"\n=== Prompt {prompt_index}, Run {run}/10 ===")
            report = measurement.run_full_experiment(
                irritability_prompt_index=prompt_index,
                questionnaire_type="all",
                method="multi_turn"
            )
            #filename = f"results/noushermes_prompt{prompt_index}_run{run+5}.json"
            #filename = f"results/gpt4o_prompt{prompt_index}_run{run+5}.json"
            filename = f"results_zero_temp/grok3mini_propmpt{prompt_index}_run{run}.json"
            #filename = f"results/claude_prompt{prompt_index}_run{run+100}.json"

            measurement.save_results(report, filename=filename)
