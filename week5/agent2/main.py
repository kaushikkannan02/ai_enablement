# from agent import agent_2

# def main():
#     print("="*60)
#     print("  Presidio Enterprise Policy Assistant")
#     print("="*60)
#     print("\nAsk about HR policies or insurance policies.")
#     print("Type 'exit' to quit.\n")
    
#     while True:
#         try:
#             user_input = input("You: ").strip()
            
#             if not user_input:
#                 continue
                
#             if user_input.lower() in {"exit", "quit", "q"}:
#                 print("\nGoodbye! 👋\n")
#                 break
            
#             # Run the agent
#             response = agent_2.run(user_input)
#             # print("Answer:::")
#             # print(response)
            
#         except KeyboardInterrupt:
#             print("\n\nGoodbye! 👋\n")
#             break
#         except Exception as e:
#             print(f"\n❌ Error: {e}\n")

# if __name__ == "__main__":
#     main()
