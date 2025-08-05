from openai import AsyncOpenAI
from typing import Dict, Any, List
from app.core.config import settings
from app.core.logging import loggers
import json

logger = loggers['openai']

class OpenAIService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        # Primary model preference list – will fall back if requested model unavailable
        self.model_preferences = [
            "gpt-4o-mini",   # OpenAI 2024-05 extremely capable, lower cost
            "gpt-4o",        # Full 4-o model if account has access
            "gpt-4-turbo",   # Older turbo model
            "gpt-4",         # Standard GPT-4
            "gpt-3.5-turbo"  # Guaranteed to exist – last-resort fallback
        ]
        self.model = self.model_preferences[0]
        logger.info("OpenAIService initialized")

    async def generate_property_insights(
        self,
        rules_output: Dict[str, Any],
        analysis_type: str = "full"
    ) -> Dict[str, Any]:
        """
        Generate expert-level insights about a property by critiquing and contextualizing the rules-based output using GPT-4.
        """
        try:
            prompt = self._create_hybrid_analysis_prompt(rules_output, analysis_type)
            response = await self._safe_chat_completion(
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            try:
                expert_insights = response.choices[0].message.content.strip()
                return {**rules_output, "expert_insights": expert_insights}
            except Exception as parse_err:
                logger.error(f"Error parsing LLM response: {parse_err}")
                return {**rules_output, "expert_insights": response.choices[0].message.content.strip()}
        except Exception as e:
            logger.error(f"Error generating property insights: {e}")
            return {**rules_output, "expert_insights": "AI insights are temporarily unavailable. Please try again later or review the metrics above for guidance."}

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for GPT-4.
        """
        return """You are an expert real estate investment analyst. Your task is to analyze property data and provide detailed, actionable, and quantitative insights about investment potential. Focus on:
1. Market positioning and value proposition
2. Investment strategy recommendations (with numbers)
3. Risk factors and mitigation strategies
4. Market trends and opportunities
5. Specific action items for investors
6. Always provide a clear bottom-line verdict (e.g., Strong Buy, Hold, Avoid) with justification

Provide your analysis in a structured format with clear section headers and bullet points. Reference the provided metrics wherever possible."""

    def _create_hybrid_analysis_prompt(self, rules_output: Dict[str, Any], analysis_type: str) -> str:
        """
        Create a prompt for the LLM that includes all rules-based metrics, scores, recommendations, and verdict,
        and asks for expert critique, context, and deeper insights.
        """
        # Compose the rules-based summary
        metrics = json.dumps({k: v for k, v in rules_output.items() if k not in ["analysis_date", "comparable_properties"]}, indent=2)
        prompt = f"""
You are a top-tier real estate investment analyst. Below is a full rules-based analysis for a property ({analysis_type} analysis):

{metrics}

Your task:
1. Critique the rules-based output: What is strong, what is missing, what would you question?
2. Add expert-level context: What would a human analyst see in these numbers that a rules engine might miss?
3. Highlight edge cases, market context, or red/yellow flags that are not obvious from the metrics alone.
4. Suggest what additional data or due diligence would be most valuable for an investor.
5. Do NOT simply repeat the numbers or recommendations above—add value, nuance, and depth.

Respond with a single, highly insightful paragraph or bullet list as "Expert Insights" for the investor.
"""
        return prompt

    def _parse_insights(self, content: str) -> Dict[str, Any]:
        """
        Parse the GPT-4 response into structured insights.
        """
        # Split the content into sections
        sections = content.split("\n\n")
        
        # Extract key components
        analysis = ""
        recommendations = []
        market_trends = {}
        risk_factors = []
        
        current_section = ""
        for section in sections:
            if "Analysis:" in section:
                current_section = "analysis"
                analysis = section.replace("Analysis:", "").strip()
            elif "Recommendations:" in section:
                current_section = "recommendations"
            elif "Market Trends:" in section:
                current_section = "market_trends"
            elif "Risk Factors:" in section:
                current_section = "risk_factors"
            elif current_section == "recommendations":
                recommendations.append(section.strip())
            elif current_section == "market_trends":
                # Parse market trends into key-value pairs
                for line in section.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        market_trends[key.strip()] = value.strip()
            elif current_section == "risk_factors":
                risk_factors.append(section.strip())
        
        return {
            "analysis": analysis,
            "recommendations": recommendations,
            "market_trends": market_trends,
            "risk_factors": risk_factors
        }

    async def get_chat_completion(self, prompt: str, model: str = None, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        try:
            response = await self._safe_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant providing real estate market analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            logger.info("Received response from OpenAI")
            return content
        except Exception as e:
            logger.error(f"Error getting OpenAI chat completion: {str(e)}")
            raise

    async def get_market_insights_and_recommendations(
        self,
        current_metrics: Dict[str, Any],
        historical_data: Dict[str, Any],
        market_analysis: Dict[str, Any],
        location: str
    ) -> Dict[str, Any]:
        """
        Generates comprehensive market insights and recommendations using OpenAI LLM.
        """
        try:
            prompt = self._construct_market_analysis_prompt(
                current_metrics, historical_data, market_analysis, location
            )
            llm_response = await self.get_chat_completion(prompt)
            
            # Attempt to parse the LLM response as JSON
            try:
                parsed_response = json.loads(llm_response)
                if "insights" not in parsed_response or "recommendations" not in parsed_response:
                    raise ValueError("LLM response missing 'insights' or 'recommendations' keys.")
                return parsed_response
            except json.JSONDecodeError:
                logger.warning("LLM response is not valid JSON. Attempting fallback parsing.")
                # Fallback: If not JSON, try to extract from raw text
                insights = """
                Based on the market data, here are some key insights:
                """ + llm_response.split("Recommendations:")[0].replace("Market Insights:", "").strip()
                recommendations = """
                Here are some recommendations based on the insights:
                """ + llm_response.split("Recommendations:")[-1].strip() if "Recommendations:" in llm_response else "No specific recommendations provided."
                
                return {
                    "insights": insights,
                    "recommendations": recommendations.split('\n') # Return recommendations as a list of strings
                }

        except Exception as e:
            logger.error(f"Error generating market insights and recommendations: {str(e)}")
            raise

    def _construct_market_analysis_prompt(
        self,
        current_metrics: Dict[str, Any],
        historical_data: Dict[str, Any],
        market_analysis: Dict[str, Any],
        location: str
    ) -> str:
        """
        Constructs a detailed prompt for the LLM based on market data.
        """
        prompt_parts = [
            f"You are a real estate market analyst. Provide a comprehensive market overview and actionable investment recommendations for {location}.",
            "Analyze the following data:",
            "\n--- Current Market Metrics ---",
            json.dumps(current_metrics, indent=2),
            "\n--- Historical Data (last few points for trends) ---",
            "Median List Price: " + str(historical_data.get('median_listing_price', {}).get('values', [])[-6:]),
            "Median Days on Market: " + str(historical_data.get('median_days_on_market', {}).get('values', [])[-6:]),
            "Price per Sqft: " + str(historical_data.get('price_per_sqft', {}).get('values', [])[-6:]),
            "\n--- ML-Generated Market Analysis ---",
            json.dumps(market_analysis, indent=2),
            "\nBased on all the provided data (current metrics, historical trends, and ML analysis), provide:",
            "1. A concise yet highly specific and detailed 'Market Insights' summary (around 3-5 sentences) highlighting key trends, market health, volatility, seasonality, and notable observations relevant to investment decisions.",
            "2. A list of 3-5 'Market Recommendations' that are highly actionable, specific, and directly derived from the analysis. Focus on concrete buying, selling, or investment strategies tailored to the current market conditions in {location}.",
            """Format your response as a JSON object with two keys: "insights" (string) and "recommendations" (list of strings). If you cannot produce JSON, provide the insights under 'Market Insights:' and recommendations under 'Recommendations:'
Example for JSON:
{
    "insights": "The {location} market is currently experiencing significant acceleration, driven by low inventory and strong buyer demand. Short-term price trends indicate a rapid upward trajectory, with moderate volatility. Seasonal analysis suggests optimal buying periods are approaching, despite overall market health being robust.",
    "recommendations": [
        "Act quickly on attractive properties due to accelerating momentum.",
        "Focus on properties with strong potential for rental yield, given the tight market conditions.",
        "Consider pre-approval for financing to strengthen offers in a competitive environment.",
        "Review historical price appreciation in specific sub-neighborhoods for targeted investments."
    ]
}
"""
        ]
        return "\n".join(prompt_parts)

    async def generate_chat_response(
        self,
        question: str,
        context: str,
        history: List[str] = []
    ) -> str:
        """
        Generate a response to a chat question using RAG (Retrieval-Augmented Generation).
        
        Args:
            question: The user's question
            context: Relevant context for answering the question
            history: List of previous questions for context
            
        Returns:
            The generated response
        """
        try:
            # Construct the system prompt
            system_prompt = """You are an expert real estate market analyst assistant. Your task is to provide detailed, accurate, and helpful responses to questions about real estate markets, properties, and investment opportunities.

Use the provided context to inform your answers. The context contains real market data and analysis that you should reference in your response.

Guidelines:
1. Be specific and data-driven in your responses
2. Reference actual numbers and metrics from the context when relevant
3. Provide actionable insights and recommendations
4. Be clear about any limitations or uncertainties
5. Maintain a professional and helpful tone

If you cannot answer a question based on the provided context, say so clearly and explain what additional information would be needed."""

            # Construct the user prompt with context and history
            user_prompt = f"""Context Information:
{context}

Previous Questions:
{chr(10).join(history) if history else 'No previous questions'}

Current Question:
{question}

Please provide a detailed response based on the context and previous questions."""

            # Generate response using GPT-4
            response = await self._safe_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating chat response: {str(e)}")
            raise

    async def _safe_chat_completion(self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 1000) -> Any:
        """Try models in preference order until one succeeds (handles 404 model_not_found)."""
        last_err: Exception | None = None
        for m in self.model_preferences:
            try:
                response = await self.client.chat.completions.create(
                    model=m,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # Update default model to first successful one for future calls
                self.model = m
                return response
            except Exception as e:
                # Only retry on model not found / 404 ; for other errors break early
                if "model_not_found" in str(e) or "does not exist" in str(e):
                    logger.warning(f"Model {m} unavailable – trying fallback")
                    last_err = e
                    continue
                else:
                    raise
        # If loop finishes without return, raise last error
        raise last_err or RuntimeError("All model fallbacks failed") 