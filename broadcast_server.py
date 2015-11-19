import SocketServer

class broadcastServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    pass