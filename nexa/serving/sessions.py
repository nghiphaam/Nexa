"""Session management for multi-user serving."""


class SessionManager:
    """Manages multiple chat sessions."""
    def __init__(self):
        self.sessions = {}

    def create_session(self, session_id):
        self.sessions[session_id] = {"history": [], "kv_cache": None}
        return self.sessions[session_id]

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def delete_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def list_sessions(self):
        return list(self.sessions.keys())
