#!/usr/bin/env python3
"""Quick test to verify the fix"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agentic.workflow import create_orchestrator

load_dotenv()

print("Testing orchestrator fix...")
print("="*70)

orchestrator = create_orchestrator()

state = {
    "messages": [HumanMessage(content="Não consigo fazer login na minha conta")],
    "ticket_metadata": {"ticket_id": "TEST-FIX", "account_id": "cultpass"}
}

config = {"configurable": {"thread_id": "TEST-FIX"}}

print("\n📝 User Query: Não consigo fazer login na minha conta")
print("\n🔄 Processing...\n")

result = orchestrator.invoke(state, config)

print("\n" + "="*70)
print("RESULT:")
print("="*70)

print(f"\n📨 Total messages: {len(result['messages'])}")
print(f"\n💬 User message: {result['messages'][0].content}")
print(f"\n🤖 Assistant response: {result['messages'][-1].content}")

if result.get("classification"):
    print(f"\n📋 Classification: {result['classification']}")

if result.get("resolution"):
    print(f"\n✅ Resolution: {result['resolution']}")

print("\n" + "="*70)
print("✅ Test completed!")
