# from utils.redis_session import RedisSessionManager
#
#
# class ChatService:
#     def __init__(self, code_agent: CodeAgent, session_manager: RedisSessionManager):
#         self.code_agent = code_agent
#         self.session_manager = RedisSessionManager(ttl=3600)
#
#     def start_task(user_input: str):
#         thread_id = str(uuid.uuid4())
#         session_state = {"messages": [HumanMessage(content=user_input)], "interrupt": ""}
#         result = code_agent.run(session_state)
#         session_state.update(result)
#         session_manager.save(thread_id, session_state)
#         return {"thread_id": thread_id, "result": result}
#
#     @chat_router.post("/feedback")
#     def feedback(thread_id: str, user_input: str):
#         session_state = session_manager.load(thread_id)
#         if not session_state:
#             return {"error": "会话已过期或不存在，请重新开始。"}
#         session_state["messages"].append(HumanMessage(content=user_input))
#         result = code_agent.run(session_state)
#         session_state.update(result)
#         session_manager.save(thread_id, session_state)
#         return {"thread_id": thread_id, "result": result}
