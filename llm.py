"""
LLM utility
Supports multiple free and paid providers with high rate limits.
Set LLM_PROVIDER in .env to switch.

Free providers with HIGH rate limits:
- "google_gemma" - Gemma 3 via Google AI Studio (14,400 requests/day) 
- "groq" - Groq with Llama 3.3 70B (14,400 requests/day)
- "cerebras" - Cerebras with Llama 3.3 70B (14,400 requests/day)
- "openrouter" - OpenRouter with multiple free models (50-1000 requests/day)
- "gemini" - Google Gemini (20 requests/day - LIMITED)

Paid providers:
- "openai" - GPT-4o (requires payment)
- "anthropic" - Claude (requires payment)
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_llm(temperature: float = 0.2):
    """
    Returns the configured LLM with HIGH rate limits.
    Set LLM_PROVIDER in .env to switch between providers.
    
    For MAXIMUM free rate limits, use:
    - "google_gemma" - 14,400 requests/day (RECOMMENDED)
    - "groq" - 14,400 requests/day
    - "cerebras" - 14,400 requests/day
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    # ===== HIGH RATE LIMIT FREE PROVIDERS (14,400 requests/day) =====
    
    if provider == "google_gemma":
        # Google Gemma 3 via Google AI Studio - 14,400 requests/day
        # Uses same API key as Gemini but with Gemma models
        try:
            from langchain_openai import ChatOpenAI
            
            api_key = os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_AI_API_KEY or GEMINI_API_KEY not found")
            
            model_name = os.getenv("GEMMA_MODEL", "gemma-3-27b-instruct")
            
            print(f"🔧 Using Google Gemma 3 model: {model_name} (14,400 requests/day)")
            
            return ChatOpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                default_headers={
                    "Content-Type": "application/json"
                }
            )
        except ImportError:
            raise ImportError("Please install langchain-openai: pip install langchain-openai")
        except Exception as e:
            print(f"❌ Error initializing Google Gemma: {e}")
            raise
    
    elif provider == "groq":
        # Groq - 14,400 requests/day with Llama 3.3 70B
        try:
            from langchain_groq import ChatGroq
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            
            print(f"🔧 Using Groq model: {model_name} (14,400 requests/day)")
            
            return ChatGroq(
                model=model_name,
                groq_api_key=api_key,
                temperature=temperature,
            )
        except ImportError:
            raise ImportError("Please install langchain-groq: pip install langchain-groq")
        except Exception as e:
            print(f"❌ Error initializing Groq: {e}")
            raise
    
    elif provider == "cerebras":
        # Cerebras - 14,400 requests/day with Llama 3.3 70B
        try:
            from langchain_openai import ChatOpenAI
            
            api_key = os.getenv("CEREBRAS_API_KEY")
            if not api_key:
                raise ValueError("CEREBRAS_API_KEY not found in environment variables")
            
            model_name = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")
            
            print(f"🔧 Using Cerebras model: {model_name} (14,400 requests/day)")
            
            return ChatOpenAI(
                base_url="https://api.cerebras.ai/v1",
                api_key=api_key,
                model=model_name,
                temperature=temperature,
            )
        except ImportError:
            raise ImportError("Please install langchain-openai: pip install langchain-openai")
        except Exception as e:
            print(f"❌ Error initializing Cerebras: {e}")
            raise
    
    # ===== MEDIUM RATE LIMIT FREE PROVIDERS =====
    
    elif provider == "openrouter":
        # OpenRouter - 50-1000 requests/day (depends on model)
        try:
            from langchain_openai import ChatOpenAI
            
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            
            model_name = os.getenv("OPENROUTER_MODEL", "google/gemma-3-27b-instruct:free")
            
            print(f"🔧 Using OpenRouter model: {model_name}")
            
            return ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key,
                model=model_name,
                temperature=temperature,
                default_headers={
                    "HTTP-Referer": os.getenv("APP_URL", "http://localhost:8501"),
                    "X-Title": "Multi-Agent Research Assistant"
                }
            )
        except ImportError:
            raise ImportError("Please install langchain-openai: pip install langchain-openai")
        except Exception as e:
            print(f"❌ Error initializing OpenRouter: {e}")
            raise
    
    elif provider == "gemini":
        # Google Gemini - 20 requests/day (LIMITED)
        try:
            from langchain_openai import ChatOpenAI
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            
            print(f"⚠️ Using Gemini model: {model_name} (ONLY 20 requests/day)")
            print("💡 For higher limits, use LLM_PROVIDER=google_gemma (14,400 requests/day)")
            
            return ChatOpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                default_headers={
                    "Content-Type": "application/json"
                }
            )
        except ImportError:
            raise ImportError("Please install langchain-openai: pip install langchain-openai")
        except Exception as e:
            print(f"❌ Error initializing Gemini: {e}")
            raise
    
    # ===== PAID PROVIDERS =====
    
    elif provider == "anthropic":
        # Anthropic Claude (PAID)
        try:
            from langchain_anthropic import ChatAnthropic
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            
            model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
            
            print(f"🔧 Using Anthropic model: {model_name} (PAID)")
            
            return ChatAnthropic(
                model=model_name,
                anthropic_api_key=api_key,
                temperature=temperature,
            )
        except ImportError:
            raise ImportError("Please install langchain-anthropic: pip install langchain-anthropic")
        except Exception as e:
            print(f"❌ Error initializing Anthropic: {e}")
            raise
    
    else:  # Default: openai (PAID)
        from langchain_openai import ChatOpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            print("⚠️ Warning: OPENAI_API_KEY not found.")
            print("💡 For FREE high-rate-limit options, set LLM_PROVIDER to one of:")
            print("   - google_gemma (14,400 req/day) - RECOMMENDED")
            print("   - groq (14,400 req/day)")
            print("   - cerebras (14,400 req/day)")
            print("   - openrouter (50-1000 req/day)")
            return MockLLM(temperature=temperature)
        
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        print(f"🔧 Using OpenAI model: {model_name} (PAID)")
        
        return ChatOpenAI(
            model=model_name,
            openai_api_key=openai_api_key,
            temperature=temperature,
        )


class MockLLM:
    """Fallback LLM for when no API keys are available."""
    def __init__(self, temperature: float = 0.2):
        self.temperature = temperature
    
    def invoke(self, messages):
        return (
            "⚠️ Mock response: No API keys configured.\n\n"
            "To use the app with FREE high-rate-limit providers:\n"
            "1. Set LLM_PROVIDER=google_gemma in .env (14,400 requests/day)\n"
            "2. Add GOOGLE_AI_API_KEY=your_key to .env\n"
            "3. Get a free API key from Google AI Studio\n\n"
            "Or try Groq: LLM_PROVIDER=groq with GROQ_API_KEY\n"
            "Or Cerebras: LLM_PROVIDER=cerebras with CEREBRAS_API_KEY"
        )
    
    def stream(self, messages):
        yield self.invoke(messages)


# Helper function to list available Google models
def list_google_models():
    """
    Helper to check which Google models you have access to.
    """
    try:
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("No Google API key found")
            return
        
        genai.configure(api_key=api_key)
        models = genai.list_models()
        
        print("📋 Available Google models:")
        print("=" * 50)
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                print(f"✅ {model.name}")
                print(f"   Methods: {', '.join(model.supported_generation_methods)}")
                print()
        
        return models
    except Exception as e:
        print(f"Error listing models: {e}")
        return None