import xml.etree.ElementTree as et
import random
import csv


def load_root(path):
    """Find the root of an xml file given a filepath (str). """
    tree = et.parse(path)
    root = tree.getroot()
    return root

def get_talks(root):
    """Get all talk elements from an xml file."""
    talks = root.findall('file')
    return talks

    
def get_talk_info(talk):
    """Extract talk information from XML element."""
    talk_id = talk.get('id')
    talkid = talk.findtext("head/talkid")
    speaker = talk.findtext("head/speaker")
    dtime = talk.findtext("head/dtime")
    keywords = talk.findtext("head/keywords")
    content = talk.findtext("content")
    url = talk.findtext("head/url")

    return talk_id,  talkid, speaker, dtime, keywords, content,  url

def info_all_talks(talks):
    """Extract talks information from XML element."""
    talk_data = []
    for talk in talks:
        talk_info = get_talk_info(talk)
        talk_data.append(talk_info)

    return talk_data

def save_to_csv(data, filename):
    """Save data to CSV file."""
    with open(filename, 'w', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Id', 'TalkId', 'Speaker', 'Dtime','Keywords', 'Content', 'URL'])
        for talk in data:
            csv_writer.writerow(talk)


def main():

    # Set the random seed for reproducibility
    random.seed(42)
    # Path of the data
    ar_df = "OUT_DIR/Arabic/arabic_talks.xml"
    en_df = "OUT_DIR/English/english_talks.xml"

    # For Arabic language
    ar_root = load_root(ar_df)
    ar_talks = get_talks(ar_root)
    ar_info_talks = info_all_talks(ar_talks)

    # Shuffle the data randomly
    random.shuffle(ar_info_talks)

    # Calculate the 80%
    split_index = int(0.8 * len(ar_info_talks))

    # Split the data into training and test sets
    train = ar_info_talks[:split_index]
    test = ar_info_talks[split_index:]

    save_to_csv(train, 'ara/train/train_arabic.csv')
    save_to_csv(test, 'ara/test/test_arabic.csv')



    # For English language
    en_root = load_root(en_df)
    en_talks = get_talks(en_root)
    en_info_talks = info_all_talks(en_talks)

    # Shuffle the data randomly
    random.shuffle(en_info_talks)

    # Calculate the 80%
    split_index = int(0.8 * len(en_info_talks))

    # Split the data into training and test sets
    train = en_info_talks[:split_index]
    test = en_info_talks[split_index:]

    save_to_csv(train, 'eng/train/train_english.csv')
    save_to_csv(test, 'eng/test/test_english.csv')



# if __name__ == "__main__":
#     main()
