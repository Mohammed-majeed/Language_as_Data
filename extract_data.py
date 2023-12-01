import xml.etree.ElementTree as et

def load_root(path):
    """Find the root of an xml file given a filepath (str). """
    tree = et.parse(path)
    root = tree.getroot()
    return root

def get_talks(root):
    """Get all talk elements from an xml file."""
    talks = root.findall('file')
    return talks

def create_xml(elements, name="output.xml"):
    """Write XML file"""
    root = et.Element("root")
    for element in elements:
        root.append(element)
    tree = et.ElementTree(root)
    tree.write(name, encoding="utf-8")
    
def corresponding_english_talks(ar_talks, en_talks):
    """Find the original talk in english """
    english_talks = []
    for ar_talk in ar_talks:
        ar_talk_id = ar_talk.findtext("head/talkid")
        for en_talk in en_talks:
            en_talk_id = en_talk.findtext("head/talkid")
            if ar_talk_id == en_talk_id:
                english_talks.append(en_talk)
    return english_talks

def main():
    # Path of the data
    ar_df = "original_data/ted_ar.xml"
    en_df = "original_data/ted_en.xml"

    # For Arabic language
    ar_root = load_root(ar_df)
    ar_talks = (get_talks(ar_root))[:300]

    # For English language
    en_root = load_root(en_df)
    en_talks = get_talks(en_root)

    # Find corresponding English talks
    english_talks = corresponding_english_talks(ar_talks, en_talks)

    # Create XML files using the create_xml function
    create_xml(english_talks, name="OUT_DIR/English/english_talks.xml")
    create_xml(ar_talks, name="OUT_DIR/Arabic/arabic_talks.xml")


# if __name__ == "__main__":
    # main()