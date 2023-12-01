import os
import extract_data as ex_data
from create_train_test_data import main as train_test_main


def make_data():
    
    # Path of the data
    ar_df = "original_data/ted_ar.xml"
    en_df = "original_data/ted_en.xml"

    # For Arabic language
    ar_root = ex_data.load_root(ar_df)
    ar_talks = (ex_data.get_talks(ar_root))[:300]

    # For English language
    en_root = ex_data.load_root(en_df)
    en_talks = ex_data.get_talks(en_root)

    # Find corresponding English talks
    english_talks = ex_data.corresponding_english_talks(ar_talks, en_talks)

    # Create XML files using the create_xml function
    ex_data.create_xml(english_talks, name="OUT_DIR/English/english_talks.xml")
    ex_data.create_xml(ar_talks, name="OUT_DIR/Arabic/arabic_talks.xml")




if __name__ == "__main__":

    if not os.path.exists('OUT_DIR'):
        os.mkdir('OUT_DIR')
        os.mkdir(os.path.join('OUT_DIR', 'Arabic'))
        os.mkdir(os.path.join('OUT_DIR', 'English'))
    
    if not os.path.exists('ara'):
        os.mkdir('ara')
        os.mkdir(os.path.join('ara','train'))
        os.mkdir(os.path.join('ara','test'))
        
    if not os.path.exists('eng'):
        os.mkdir(os.path.join('eng'))
        os.mkdir(os.path.join('eng','train'))
        os.mkdir(os.path.join('eng','test'))


    make_data()
    train_test_main()