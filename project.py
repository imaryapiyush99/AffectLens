from affectlens.preprocessing import remove_stopwords, preprocess, clean_text, lemmatize, tokenize, get_processed_text

def main() -> None:
    print("AffectLens — ready.")
    print(get_processed_text("HTML &amp; entities"))

if __name__ == "__main__":
    main()