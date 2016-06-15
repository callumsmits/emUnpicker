#!/usr/bin/env python2.7
from sys import argv
import sys
import os
import math
from os import listdir
from os.path import isfile, join, abspath
import random

def main():
    try:
        allPickRootName = str(sys.argv[1])
        correctPickRootName = str(sys.argv[2])
        outputRootName = str(sys.argv[3])
    except:
        print '\nUsage: trainingSetFromStarFilesSearchByCorrect.py allPickRootName correctPickRootName outputRootName\n'
        raise SystemExit

    random.seed()

#   Get list of Images for source of training material
    MRCFilesToCheck = []
    allFiles = [ f for f in listdir(os.getcwd()) if f.endswith(correctPickRootName) ]
    for file in allFiles:
        MRCName = file[0:file.find(correctPickRootName)] + '.mrc'
        MRCFilesToCheck.append(MRCName)

    for MRCFile in MRCFilesToCheck:
        allPickInputName = MRCFile[0:MRCFile.find('.mrc')] + allPickRootName
        correctPickInputName = MRCFile[0:MRCFile.find('.mrc')] + correctPickRootName
        correctPicks = []
        with open(correctPickInputName, 'r') as pickInputFile:
            for l in range(0,9):
                pickInputFile.readline()
            for line in pickInputFile:
                if line.strip():
                    correctPicks.append(line)

        outputString = ''
        badPicks = []
        with open(allPickInputName, 'r') as pickInputFile:
            for l in range(0,9):
                outputString += pickInputFile.readline()
            for line in pickInputFile:
                if line.strip():
                    if not any(line in s for s in correctPicks):
                        badPicks.append(line)

        if (len(badPicks) == 0):
            print 'No unpicks in that file! Can\'t use for training'
            continue
        for particle in correctPicks:
            outputString += particle[0:len(particle)-1] + ' P\n'
            chosen_line = badPicks[random.randint(0,len(badPicks)-1)]
            outputString += chosen_line[0:len(chosen_line)-1] + ' N\n'


        pickOutputName = MRCFile[0:MRCFile.find('.mrc')] + outputRootName
        with open(pickOutputName, 'w') as outputFile:
            outputFile.write(outputString)

if __name__ == '__main__':
    main()
