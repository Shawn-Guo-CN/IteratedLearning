import sys
sys.path.insert(0, 'E:\\GitWS\\IteratedLearning\\')

from IteratedLearning import ImageReferentialIterator


def test_image_referential_iterator():
    iterator = ImageReferentialIterator('E:\\GitWS\\IteratedLearning\\tests\\tmp\\img_set_25', 32)
    
    for idx, batch in enumerate(iterator):
        print('batch idx:', idx)
    
    return True


if __name__ == '__main__':
    test_image_referential_iterator()
