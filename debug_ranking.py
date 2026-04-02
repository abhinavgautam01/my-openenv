#!/usr/bin/env python3
"""
Debug script to test ranking task step by step.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import EmailTriageEnv, get_llm_ranking_decision, create_ranking_action, format_emails_for_ranking_prompt
from openai import OpenAI

async def debug_ranking():
    """Debug ranking task step by step."""
    print("=== DEBUGGING RANKING TASK ===")
    
    # Initialize OpenAI client
    client = OpenAI(
        base_url=os.getenv("API_BASE_URL"),
        api_key=os.getenv("HF_TOKEN")
    )
    print(f"✅ OpenAI client initialized")
    
    # Create environment
    env = await EmailTriageEnv.from_docker_image(
        os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
    )
    print(f"✅ Environment created")
    
    try:
        # Reset environment
        result = await env.reset(task_type="ranking", seed=42)
        print(f"✅ Environment reset: {len(result.observation.emails)} emails")
        
        emails = result.observation.emails
        print(f"📧 Email IDs: {[e.id for e in emails]}")
        
        # Create prompt
        user_prompt = format_emails_for_ranking_prompt(emails)
        print(f"📝 Prompt created (length: {len(user_prompt)})")
        
        # Get LLM decision
        print("🤖 Calling LLM...")
        decision = get_llm_ranking_decision(client, emails, "ranking", user_prompt)
        print(f"🤖 LLM decision: {decision}")
        
        # Create action
        action = create_ranking_action(decision)
        print(f"⚡ Action created: email_id={action.email_id}, ranking={action.ranking}")
        
        # Send action to environment
        print("🚀 Sending action to environment...")
        result = await env.step(action)
        
        print(f"✅ Step result: done={result.done}, reward={result.reward}")
        print(f"📋 Last action result: {result.info.get('last_action_result', 'None')}")
        print(f"🏆 Final score: {result.info.get('final_score', 'None')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await env.close()
        print("✅ Environment closed")

if __name__ == "__main__":
    asyncio.run(debug_ranking())
