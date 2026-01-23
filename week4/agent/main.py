# from langchain_core.messages import HumanMessage
# from agent.agent import policy_agent
# def main():
#     print("Enterprise Policy Assistant (type 'exit')")
#     while True:
#         q = input("\n> ")
#         if q.lower() in {"exit", "quit"}:
#             break

#         result = policy_agent.invoke({
#             "messages": [HumanMessage(content=q)]
#         })

#         print("\nAnswer:")
#         print(result["messages"][-1].content)

# if __name__ == "__main__":
#     main()

from agent.agent import policy_agent

def main():
    print("="*60)
    print("  Presidio Enterprise Policy Assistant")
    print("="*60)
    print("\nAsk about HR policies or insurance policies.")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in {"exit", "quit", "q"}:
                print("\nGoodbye! 👋\n")
                break
            
            # Run the agent
            response = policy_agent.run(user_input)
            # print("Answer:::")
            # print(response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()
