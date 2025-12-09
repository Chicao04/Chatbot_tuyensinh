import os
from ragbot import get_settings, RAGChatbot

def main():
    s = get_settings()
    bot = RAGChatbot(s)

    for grade in ["chuyennganh", "diem_chuan", "thacsi_tiensi"]:
        grade_dir = os.path.join(s.input_dir, grade)
        print("Indexing:", grade_dir)
        bot._get_index(grade)

    print("Done. Cache at:", s.cache_dir)

if __name__ == "__main__":
    main()
