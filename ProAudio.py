#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:25:21 2017

@author: 390771
"""
import os
import pydub
import python_speech_features as psf
import numpy as np

class Pretreatment():
    def __init__(self,audioPath,outputPath):
        self.__audioPath = audioPath
        self.__outputPath = outputPath
        self.__audioPathlist = os.listdir(audioPath)

    def getAudioPath(self):
        audioPath = self.__audioPath
        return audioPath

    def getOutputPath(self):
        print self.__outputPath

    def writeToTxt(self,list_name,file_path):
        fp = open(file_path, "w+")
        for item in list_name:
            fp.write(str(item) + "\n")  # list中一项占一行
        fp.close()

    def exactMFCC(self,fileName):
        pathName = os.path.join(self.__audioPath,fileName)
        if fileName.find(".mp3") != -1 or fileName.find(".MP3") != -1:
            try:
                sounds = pydub.AudioSegment.from_mp3(pathName)
            except:
                return 'audio is damaged !'
        elif fileName.find(".wav") != -1 or fileName.find(".WAV") != -1:
            try:
                sounds = pydub.AudioSegment.from_wav(pathName)
            except:
                return 'audio is damaged !'
        else :
            return 'reading audio is error !'
        datas = sounds.get_array_of_samples()
        frame_rate = sounds.frame_rate
        # datasArrary = np.array(datas.tolist())
        datasArrary = np.array(datas)
        psMFCC = psf.mfcc(datasArrary,frame_rate)
        temp = psMFCC.tolist()
        file_path = os.path.join(self.__outputPath,fileName)
        if fileName.find(".mp3") != -1 or fileName.find(".MP3") != -1:
            outPutMFCCFile = fileName.replace(".mp3", '.txt')
            outPutMFCCFile = outPutMFCCFile.replace(".MP3", '.txt')
        elif fileName.find(".wav") != -1 or fileName.find(".WAV") != -1:
            outPutMFCCFile = fileName.replace(".wav", '.txt')
            outPutMFCCFile = outPutMFCCFile.replace(".WAV", '.txt')
        else :
            return 'audio format is error !'
        outPutMFCCFile_path = os.path.join(self.__outputPath,outPutMFCCFile)
        # print 'The fileName is ', outPutMFCCFile_path
        self.writeToTxt(temp,outPutMFCCFile_path)
        print 'writeToTxt suc !'
        return psMFCC

    def catMFCCFile(self):
        arr = np.array(range(13))
        for audioName in self.__audioPathlist:
           if audioName.find(".mp3") != -1 or audioName.find(".MP3") != -1 or audioName.find(".WAV") != -1 or audioName.find(".wav") != -1:
               psMFCC = self.exactMFCC(audioName)

               if type(psMFCC) == str:
                   continue
               print 'psMFCC type is', type(psMFCC), psMFCC.shape
               arr = np.vstack((psMFCC,arr))
           else:
               continue
        outPutMFCCFileTotalpath = os.path.join(self.__outputPath, 'finel_MFCC.txt')
        self.writeToTxt(arr, outPutMFCCFileTotalpath)
        print 'write suc !'
        return arr

    # 提取emd模态分解得到能量特征函数以及互关系数
    def exactEMD(self,fileName):
        import exactEnergyAndCor as exact
        pathName = os.path.join(self.__audioPath, fileName)
        if fileName.find(".mp3") != -1 or fileName.find(".MP3") != -1:
            try:
                signal = exact.readAudioData(pathName)
            except:
                return 'audio is damaged !'
        elif fileName.find(".wav") != -1 or fileName.find(".WAV") != -1:
            try:
                signal = exact.readAudioData(pathName)
            except:
                return 'audio is damaged !'
        else :
            return 'audio is damaged !'
        imfs = exact.emd(signal,n_imfs=10)
        if type(imfs) == str:
            return 'emd error !'
        feature = exact.exactEnergyAndCor(imfs,signal)
        # print imfs.size,feature.shape
        # print feature
        if fileName.find(".mp3") != -1 or fileName.find(".MP3") != -1:
            outPutEMDFile = fileName.replace(".mp3", '.txt')
            outPutEMDFile = outPutEMDFile.replace(".MP3", '.txt')
        elif fileName.find(".wav") != -1 or fileName.find(".WAV") != -1:
            outPutEMDFile = fileName.replace(".wav", '.txt')
            outPutEMDFile = outPutEMDFile.replace(".WAV", '.txt')
        else :
            return 'audio format is error !'
        outPutEMDFile_path = os.path.join(self.__outputPath,outPutEMDFile)
        # print 'The fileName is ', outPutMFCCFile_path
        self.writeToTxt(feature,outPutEMDFile_path)
        print 'writeToTxt suc !'
        return feature

    def catEMDFile(self):
        arr = np.array(range(22))
        for audioName in self.__audioPathlist:
           if audioName.find(".mp3") != -1 or audioName.find(".MP3") != -1 or audioName.find(".WAV") != -1 or audioName.find(".wav") != -1:
               psEMD = self.exactEMD(audioName)
               if type(psEMD) == str:
                   continue
               arr = np.vstack((psEMD,arr))
           else:
               continue
        # audioName = self.__audioPathlist[0]
        # if audioName.find(".mp3") != -1 or audioName.find(".MP3") != -1 or audioName.find(".WAV") != -1 or audioName.find(".wav") != -1:
        #     psEMD = self.exactEMD(audioName)
        #     if type(psEMD) == str:
        #         return 'error'
        #     arr = np.vstack((psEMD,arr))
        # else:
        #     return 'error '
        outPutEMDFileTotalpath = os.path.join(self.__outputPath, 'finel_EMD.txt')
        self.writeToTxt(arr, outPutEMDFileTotalpath)
        print 'write suc !'
        return arr


if __name__ == "__main__":
    # 二分类提取MFCC特征
    path = r'/home/390771/Data/audiolib/norNoise'
    outputPath = r'/home/390771/cc/norNoise'
    p1 = Pretreatment(path,outputPath)
    norP = p1.catMFCCFile()
    path = r'/home/390771/Data/audiolib/errNoise'
    outputPath = r'/home/390771/cc/errNoise'
    p2 = Pretreatment(path,outputPath)
    errP = p2.catMFCCFile()
    #result
    resul = np.vstack((norP,errP))
   #save as dirct  in the current directory
    res = dict()
    res['title'] = '=======MFCC_feature==========='
    res['data'] = resul
    res['taget'] = np.vstack((np.ones([len(norP),1]),np.zeros([len(errP),1])))
    import pickle
    output = open('hh1.pkl', 'wb')
    pickle.dump(res, output)
    output.close()
    # # 二分类提取EMD本征函数能量比以及互关系数
    # path = r'/home/390771/Data/audiolib/norNoise'
    # outputPath = r'/home/390771/emd_feature/norNoise'
    # p3 = Pretreatment(path, outputPath)
    # norEMDP = p3.catEMDFile()
    # print 'final',norEMDP
    # path = r'/home/390771/Data/audiolib/errNoise'
    # outputPath = r'/home/390771/emd_feature/errNoise'
    # p4 = Pretreatment(path, outputPath)
    # errEMDP = p4.catMFCCFile()
    # # result
    # resul_EMD = np.vstack((norEMDP, errEMDP))
    # save as dirct  in the current directory
    # res_emd = dict()
    # res_emd['title'] = '=======MFCC_feature==========='
    # res_emd['data'] = resul_EMD
    # res_emd['taget'] = np.vstack((np.ones([len(norEMDP), 1]), np.zeros([len(errEMDP), 1])))
    # import pickle
    #
    # outputEMD = open('emdfeature.pkl', 'wb')
    # pickle.dump(res_emd, outputEMD)
    # outputEMD.close()