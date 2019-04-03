__author__ = 'aagrawal'
__version__ = '0.9'

# Interface for accessing the VQA dataset.

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
# (https://github.com/pdollar/coco/blob/master/PythonAPI/pycocotools/coco.py).

# The following functions are defined:
#  VQA        - VQA class that loads VQA annotation file and prepares data structures.
#  getQuesIds - Get question ids that satisfy given filter conditions.
#  getImgIds  - Get image ids that satisfy given filter conditions.
#  loadQA     - Load questions and answers with the specified question ids.
#  showQA     - Display the specified questions and answers.
#  loadRes    - Load result file and create result object.

# Help on each function can be accessed by: "help(COCO.function)"

import json
import datetime
import copy
import cv2
import numpy as np
import tensorflow as tf


class VQA:
    def __init__(self, annotation_file=None,
                 question_file=None, complement_file=None):
        """
           Constructor of VQA helper class for reading and
           visualizing questions and answers.
        param:
         annotation_file (str): location of VQA annotation file

        return:
        """
        # load dataset
        self.questions = {}
        self.dataset = {}
        self.complements = {}
        self.qa = {}
        self.qqa = {}
        self.imgToQA = {}
        if annotation_file is not None and question_file is not None and complement_file is not None:
            print('loading VQA annotations and questions into memory...')
            time_t = datetime.datetime.utcnow()
            with open(annotation_file, 'r') as annFile:
                dataset = json.load(annFile)
            with open(question_file, 'r') as quesFile:
                questions = json.load(quesFile)
            with open(complement_file, 'r') as compFile:
                complements = json.load(compFile)
            print(datetime.datetime.utcnow() - time_t)
            annFile.close()
            quesFile.close()
            compFile.close()
            self.dataset = dataset
            self.questions = questions
            self.complements = dict(complements)
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        imgToQA = {ann['image_id']: [] for ann in self.dataset['annotations']}
        qa = {ann['question_id']: [] for ann in self.dataset['annotations']}
        qqa = {ann['question_id']: [] for ann in self.dataset['annotations']}
        for ann in self.dataset['annotations']:
            imgToQA[ann['image_id']] += [ann]
            qa[ann['question_id']] = ann
        for ques in self.questions['questions']:
            qqa[ques['question_id']] = ques
        qqaCopy = copy.deepcopy(qqa)
        for qid, ques in qqaCopy.items():
            try:
                compIndex = self.complements[qid]
                # print(compIndex)
                qqa[qid]['complement_img'] = qqa[compIndex]['image_id']
            except:
                qqa.pop(qid)
        print('Trainind data size: ' + str(len(qqa.keys())))
        print('index created!')

        # create class members
        self.qa = qa
        self.qqa = qqa
        self.imgToQA = imgToQA

    def info(self):
        """
        Print information about the VQA annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('%s: %s' % (key, value))

    def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
        """
        Get question ids that satisfy given filter conditions.
        default skips that filter
        :param     imgIds    (int array)   : get question ids for given imgs
                quesTypes (str array)   : get question ids for
                                          given question types
                ansTypes  (str array)   : get question ids for
                                          given answer types
        :return:    ids   (int array)   : integer array of question ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                anns = sum([self.imgToQA[imgId] for imgId in
                            imgIds if imgId in self.imgToQA], [])
            else:
                anns = self.dataset['annotations']
            anns = anns if len(quesTypes) == 0 else [
                ann for ann in anns if ann['question_type'] in quesTypes]
            anns = anns if len(ansTypes) == 0 else [
                ann for ann in anns if ann['answer_type'] in ansTypes]
        ids = [ann['question_id'] for ann in anns]
        return ids

    def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
        """
        Get image ids that satisfy given filter conditions.
        default skips that filter
        :param quesIds   (int array)   : get image ids for given question ids
               quesTypes (str array)   : get image ids for given question types
               ansTypes  (str array)   : get image ids for given answer types
        :return: ids     (int array)   : integer array of image ids
        """
        quesIds = quesIds if type(quesIds) == list else [quesIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(quesIds) == 0:
                anns = sum([self.qa[quesId] for quesId in
                            quesIds if quesId in self.qa], [])
            else:
                anns = self.dataset['annotations']
            anns = anns if len(quesTypes) == 0 else [
                ann for ann in anns if ann['question_type'] in quesTypes]
            anns = anns if len(ansTypes) == 0 else [
                ann for ann in anns if ann['answer_type'] in ansTypes]
        ids = [ann['image_id'] for ann in anns]
        return ids

    def loadQA(self, ids=[]):
        """
        Load questions and answers with the specified question ids.
        :param ids (int array)       : integer ids specifying question ids
        :return: qa (object array)   : loaded qa objects
        """
        if type(ids) == list:
            return [self.qa[id] for id in ids]
        elif type(ids) == int:
            return [self.qa[ids]]

    def showQA(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            quesId = ann['question_id']
            print("Question: %s" % (self.qqa[quesId]['question']))
            for ans in ann['answers']:
                print("Answer %d: %s" % (ans['answer_id'], ans['answer']))

    def loadRes(self, resFile, quesFile):
        """
        Load result file and return a result object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = VQA()
        res.questions = json.load(open(quesFile))
        res.dataset['info'] = copy.deepcopy(self.questions['info'])
        res.dataset['task_type'] = copy.deepcopy(self.questions['task_type'])
        res.dataset['data_type'] = copy.deepcopy(self.questions['data_type'])
        res.dataset['data_subtype'] = copy.deepcopy(self.questions['data_subtype'])
        res.dataset['license'] = copy.deepcopy(self.questions['license'])

        print('Loading and preparing results...     ')
        time_t = datetime.datetime.utcnow()
        anns = json.load(open(resFile))
        assert type(anns) == list, 'results is not an array of objects'
        annsQuesIds = [ann['question_id'] for ann in anns]
        assert set(annsQuesIds) == set(self.getQuesIds()), \
            'Results do not correspond to current VQA set.\
            Either the results do not have predictions for all question ids \
            in annotation file or there is atleast one question id that does \
            not belong to the question ids in the annotation file.'
        for ann in anns:
            quesId = ann['question_id']
            if res.dataset['task_type'] == 'Multiple Choice':
                assert ann['answer'] in self.qqa[quesId]['multiple_choices'], 'predicted answer is not one of the multiple choices'
            qaAnn = self.qa[quesId]
            ann['image_id'] = qaAnn['image_id']
            ann['question_type'] = qaAnn['question_type']
            ann['answer_type'] = qaAnn['answer_type']
        print('DONE (t=%0.2fs)'
              % ((datetime.datetime.utcnow() - time_t).total_seconds()))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res


class TFDataLoaderUtil:

    def __init__(self, data_dir, dataSubType):

        self.dataSubType = dataSubType
        self.dataDir = data_dir
        # this should be '' when using VQA v2.0 dataset
        self.versionType = 'v2_'
        # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
        self.taskType = 'OpenEnded'
        # 'mscoco' only for v1.0. 'mscoco' for
        # real and 'abstract_v002' for abstract for v1.0.
        self.dataType = 'mscoco'
        self.annFile = '%s/Annotations/%s%s_%s_annotations.json' % (
            self.dataDir,
            self.versionType,
            self.dataType,
            dataSubType)
        self.quesFile = '%s/Questions/%s%s_%s_%s_questions.json' % (
            self.dataDir, self.versionType,
            self.taskType, self.dataType, dataSubType)
        self.imgDir = '%s/Images/%s/' % (self.dataDir, dataSubType)
        self.complementFile = '%s/Complementary_pairs/%s%s_%s_complementary_pairs.json' % (self.dataDir,
                                                                                           self.versionType,
                                                                                           self.dataType,
                                                                                           dataSubType)

        try:
            self.vqa = VQA(self.annFile, self.quesFile, self.complementFile)
        except Exception as e:
            raise IOError(str(e) +
                          ". Unable to find training files.\
                          question_location='"
                          + self.quesFile +
                          "'. Image_location='"+self.imgDir+"'.")

        self.dataset = self.vqa.qqa

    def genDataBatchesIds(self, allQuestionIds=None, BATCH_SIZE=256):
        if allQuestionIds is None:
            allQuestionIds = list(self.dataset.keys())

        allQuestionIds = np.array(allQuestionIds)
        np.random.shuffle(allQuestionIds)
        num_batches = int(allQuestionIds.shape[0] / BATCH_SIZE)
        batchDataIds = np.split(allQuestionIds[:num_batches*BATCH_SIZE],
                                num_batches)

        if (allQuestionIds.shape[0] % BATCH_SIZE) > 0:
            batchDataIds.append(np.array(
                allQuestionIds[num_batches*BATCH_SIZE:]))

        return batchDataIds

    def getQuesImageCompTriplets(self, vqaDatasetIds=[]):
        if len(vqaDatasetIds) <= 0:
            _vqaDataset = self.dataset
        else:
            vqaDatasetIds = set(vqaDatasetIds)
            _vqaDataset = {key: value for key, value in
                           self.dataset.items() if key in vqaDatasetIds}

        return list((key, value['image_id'], value['complement_img'])
                    for key, value in _vqaDataset.items())

    def dataLoaderFromDataIds(self, dataTriplets=None,
                              imageResize=(448, 448, 3)):

        _dataTriplets = dataTriplets or self.getQuesImageCompTriplets()
        quesSet = []
        imgSet = []
        compImgSet = []
        # print(len(_dataTriplets))
        for item in _dataTriplets:
            quesSet.append(self.dataset[item[0]]['question'])

            loadedImg = cv2.resize(cv2.imread(
                    self.imgDir + 'COCO_' + self.dataSubType + '_' +
                    str(item[1]).zfill(12) + '.jpg'),
                    imageResize[:2]).astype(np.float32)

            compImg = cv2.resize(cv2.imread(
                    self.imgDir + 'COCO_' + self.dataSubType + '_' +
                    str(item[2]).zfill(12) + '.jpg'),
                    imageResize[:2]).astype(np.float32)

            imgSet.append(loadedImg)
            compImgSet.append(compImg)

        return (np.asarray(quesSet),
                np.asarray(imgSet) / 255,
                np.asarray(compImgSet) / 255)

    def genTFDatasetObject(self, tokenizer, MAX_LEN, BATCH_SIZE,
                           NUM_PARALLEL_CALLS, BUFFER_SIZE):

        dataTriplets = self.getQuesImageCompTriplets()
        tfDataset = tf.data.Dataset.from_tensor_slices(dataTriplets)
        tfDataset = tfDataset.batch(BATCH_SIZE)
        tfDataset = tfDataset.map(tf.py_func(
                lambda x: self.DataLoaderFromDataIds(tokenizer, MAX_LEN, x),
                list(dataTriplets),
                (tf.float32, tf.float32, tf.float32)),
                num_parallel_calls=NUM_PARALLEL_CALLS)
        tfDataset.prefetch(buffer_size=BUFFER_SIZE)

        return tfDataset.make_one_shot_iterator()
