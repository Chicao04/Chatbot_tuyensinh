from ragbot_tuyensinh import get_settings, RAGChatbot

def main():
    s = get_settings()
    bot = RAGChatbot(s)

    print("RAG Tuyển sinh chatbot (chuyennganh/diem_chuan/thacsi_tiensi). Gõ 'exit' để thoát.\n")
    while True:
        q = input("Bạn: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        out = bot.answer(q)
        print(f"\n[Route] {out['grade']}")
        print(out["answer"])
        print("\n---\n")

if __name__ == "__main__":
    main()
