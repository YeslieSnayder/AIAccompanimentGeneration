import math
import os
import sys

import mido
import numpy as np

NOTE_COUNT = 128

# You can choose any parameters as you want
GENERATIONS = 200
POPULATION_SIZE = 30
MUTATION_CHANCE = 0.4
MUTATION_ACCORD_CHANCE = 0.3
MUTATION_EMPTY_ACCORD_CHANCE = 0.01
CROSSOVER_NUMBER = 20


class Note:
    """
    Class for notes in accompaniment on each moment of time.
    """
    def __init__(self, elements=None, minimal=24, maximal=60):
        """
        Constructor of Note.
        :param elements: pre-defined list of notes for the particular time in accompaniment.
            If the parameter is None, then creates new numpy array containing random number of given domain.
        :param minimal: define minimal boundary for randomly created notes.
        :param maximal: define maximal boundary for randomly created notes.
        """
        if elements is None:
            self.notes = np.random.randint(minimal, maximal, 3)
            for i in range(self.notes.shape[0]):
                if self.notes[i] in self.notes[:i]:
                    self.notes[i] = max(self.notes[:i]) + 2
        else:
            self.notes = elements

    def crossover(self, other):
        """
        Crossover for notes. Takes that note and goes to the note in parameter.
        :param other: instance of Note. Other note to move on.
        :return: new set of crossovered notes.
        """
        if np.random.uniform(0, 1, 1) < 0.1 or self.notes.size == 0 or other.notes.size == 0:
            return Note(self.notes)
        notes = np.zeros(shape=self.notes.shape, dtype=np.int32)
        for n in range(self.notes.shape[0]):
            notes[n] = self.notes[n] - ((self.notes[n] - other.notes[n]) // 2)
            if notes[n] in notes[:n]:
                notes[n] = max(notes) + 2
            if notes[n] >= 127:
                notes[n] = min(notes) + 2
        return Note(notes)

    def mean(self):
        """
        Calculates mean value of all notes of particular time (of that object).
        :return: mean value of self.notes.
        """
        if self.notes.size == 0:
            return 0.
        return self.notes.sum() / self.notes.shape[0]


class Accompaniment:
    """
    Class for accompaniment. Contains notes of the accompaniment.
    """
    def __init__(self, size, duration, notes=None):
        """
        Constructor of Accompaniment.
        :param size: length of the accompaniment.
        :param duration: duration of one sound in accompaniment.
            All sounds are of the same duration.
        :param notes: pre-defined list of notes for accompaniment.
        """
        self.size = size
        self.duration = duration
        if notes is None:
            self.notes = [Note() for _ in range(size)]
        else:
            self.notes = notes

    def fitness(self, music):
        """
        Fitness function for genetic algorithm.
        There is some magic that is described in report.
        :param music: initial music in data format.
        :return: fitness of that accompaniment. Lower is better.
        """
        big_distance = 0.
        big_shift = 0.
        many_holes = 0.
        above_music = 0.
        in_accord = 0.
        module = 0.

        prev = None
        min_music = None
        allowed_notes = {}
        for t in range(music.shape[0]):
            if min_music is None or min(music[t]) < min_music:
                min_music = min(music[t])
        for i, note in enumerate(self.notes):
            if prev is not None:
                big_shift += abs(prev.mean() - note.mean())
            n = note.notes
            if n.size == 0:
                many_holes += 1
            else:
                prev = note
                distances = [abs(n[0] - n[1]), abs(n[0] - n[2]), abs(n[1] - n[2])]
                if any(distances) > 2:
                    big_distance += sum(distances)
                if any(distances) == 1:
                    big_distance += 10
            music_notes = []
            for t in range(int(i * self.duration), int((i + 1) * self.duration)):
                if t >= music.shape[0]:
                    break
                for nn in range(music.shape[1]):
                    if music[t][nn] > 0:
                        music_notes.append(nn)
                        allowed_notes[nn] = True
                        break
            if len(n) > 0 and len(music_notes) > 0:
                mean_music = sum(music_notes) / len(music_notes)
                mean_acc = sum(n) / len(n)
                if mean_acc > mean_music:
                    above_music += mean_acc - mean_music
                for nn in n:
                    temp = 0
                    for mn in music_notes:
                        if abs(nn - mn) % 12 == 0:
                            temp = 0
                            break
                        if abs(nn - mn) % 12 > temp:
                            temp = abs(nn - mn) % 12
                    if temp > 0:
                        module += 10 * temp
            if len(n) > 0:
                if max(n) > min_music - 12:
                    in_accord += abs(max(n) - min_music)
                if min(n) < min_music - 32:
                    in_accord += abs(min_music - min(n))
                for nn in n:
                    temp = 0
                    for k in allowed_notes:
                        if abs(nn - k) % 12 == 0:
                            temp = 0
                            break
                        if abs(nn - k) % 12 > temp:
                            temp = abs(nn - k) % 12
                    if temp > 0:
                        module += temp

        return 10 * big_distance + \
               60 * big_shift + \
               500 * many_holes + \
               1 * above_music + \
               5 * in_accord + \
               10 * module

    def crossover(self, other):
        """
        Crossover for that accompaniment and parameter (also Accompaniment).
        :param other: instance of the Accompaniment.
        :return: crossovered accompaniment.
        """
        notes = []
        for t in range(self.size):
            notes.append(self.notes[t].crossover(other.notes[t]))
        return Accompaniment(self.size, self.duration, notes)

    def mutate(self):
        """
        Mutation of the Accompaniment. Creates new instance of Accompaniment.
        :return: mutated accompaniment.
        """
        notes = []
        for t in range(self.size):
            if np.random.uniform(0, 1, 1) < MUTATION_ACCORD_CHANCE:
                notes.append(Note())
            elif np.random.uniform(0, 1, 1) < MUTATION_EMPTY_ACCORD_CHANCE:
                notes.append(Note(np.array([])))
            else:
                notes.append(self.notes[t])
        return Accompaniment(self.size, self.duration, notes)

    def toNPArray(self, music, need_more):
        """
        Converts instance of Accompaniment to numpy array.
        Creates new numpy array containing initial music with accompaniment.
        :param music: initial music in form of data used in program.
        :param need_more: Boolean. Some music can end up with accord with duration less than half of tact.
            In that case, we need to add accompaniment at the end.
        :return: numpy array containing initial music with accompaniment.
        """
        result = music
        if need_more:
            size = int(self.size)
        else:
            size = int(self.size - self.duration)
        for t in range(0, size, int(self.duration)):
            if self.notes[t].notes.size != 0:
                note = self.notes[t].notes
                for i in range(note.shape[0]):
                    if result[t][note[i]] == 0:
                        result[t][note[i]] = self.duration
        return result


def printNotes(data, score=None):
    """
    Prints music.
    :param data: Music representation in data used through program.
    :param score: Fitness score of that song.
    """
    if score is not None:
        print(f'Score: {score}')
        return
    print('---------------------------')
    print('---   BEST POPULATION   ---')
    if score is not None:
        print(f'---   SCORE: {score}   ---')
    for note in range(data.shape[1], -1, -1):
        if note < 35 or note > 73:
            continue
        print(note, end=':')
        for t in range(data.shape[0]):
            if data[t][note] == 0:
                print(' ', end='')
            else:
                print(int(data[t][note]), end='')
        print()
    print('---------------------------')


def getPopulation(population_size, time_length, accompaniment_length):
    """
    Creates new population of given size.
    :param population_size: size of population.
    :param time_length:
    :param accompaniment_length:
    :return:
    """
    population = [Accompaniment(time_length, accompaniment_length) for _ in range(population_size)]
    return population


def replacePopulation(data, pop_old, pop_evolved, population_size):
    """
    Replace population with new one by combining old and new populations.
    :param data: Initial music data.
    :param pop_old: Old population from previous step.
    :param pop_evolved: Population after crossover and mutation.
    :param population_size: Number of accompaniments in new population.
    :return: Population with best fitness.
    """
    populations = pop_old + pop_evolved
    fit = [p.fitness(data) for p in populations]
    p_fit = list(zip(fit, populations))
    p_fit = sorted(p_fit, key=lambda x: x[0])
    populations = [p[1] for p in p_fit]
    return populations[:population_size]


def crossover(population, num=CROSSOVER_NUMBER):
    """
    Makes crossover over population.
    :param population: list of Accompaniments.
    :param num: amount of crossovers. Specifies size of the output list.
    :return: list of crossovered population.
    """
    result = []
    for i in range(num):
        indices = np.random.randint(0, len(population), 2)
        result.append(population[indices[0]].crossover(population[indices[1]]))
    return result


def mutate(population):
    """
    Mutation function. Do mutation on a population with given probability.
    :param population: list of Accompaniments.
    :return: list of new population including mutated and not mutated.
    """
    mutation = []
    for p in population:
        if np.random.uniform(0, 1, 1) < MUTATION_CHANCE:
            mutation.append(p.mutate())
        else:
            mutation.append(p)
    return mutation


def evolution(data, accompaniment_length, add_sound, generations=GENERATIONS, population_size=POPULATION_SIZE):
    """
    Main function to start evolutionary algorithm for accompaniment generation.
    :param data: Initial music that will be used to generate accompaniment for.
    :param accompaniment_length: Duration of the accompaniment in terms of atomic sound in the music.
    :param add_sound: Boolean parameter to increase total duration of the music.
    :param generations: Number of generations.
    :param population_size: Number of accompaniments in one population.
    :return: Music with accompaniment.
    """
    population = getPopulation(population_size, data.shape[0], accompaniment_length)

    for g in range(generations):
        evol = crossover(population)
        evol = mutate(evol)
        population = replacePopulation(data, population, evol, population_size)
        print(f'Generation: {g + 1}\tFitness: {population[0].fitness(data)}')
    data = population[0].toNPArray(data, add_sound)
    return data


def messages2data(messages, atomic, length):
    """
    Converts list of mido.Message to numpy array that will be used in genetic algorithm.
    Shape of the output array depends on parameters atomic and length.
    :param messages: list of mido.Message.
    :param atomic: minimal length of note (duration).
    :param length: duration of the whole music.
    :return: numpy.array that represents beginning and duration of notes in music.
    """
    data = np.zeros(((length // atomic) + 1, NOTE_COUNT))
    cur = 0
    cur_notes = {}
    for msg in messages:
        if msg.type == 'note_on':
            cur_notes[msg.note] = cur + (msg.time // atomic)
        else:
            data[cur_notes[msg.note]][msg.note] = (cur + (msg.time // atomic)) - cur_notes[msg.note]
            del cur_notes[msg.note]
        cur += msg.time // atomic
    return data


def data2messages(data, atomic):
    """
    Converts numpy.array object to list of mido.Message.
    :param data: numpy.array that represents beginning and duration of notes in music.
    :param atomic: minimal duration of note.
    :return: list of mido.Message.
    """
    messages = []
    notes_on = []
    last_event = 0
    for t in range(data.shape[0]):
        i = 0
        while i < len(notes_on):
            note = notes_on[i]
            if t == note[1] + note[2]:
                time = int((t - last_event) * atomic)
                messages.append(mido.Message(type='note_off', note=note[0], velocity=0, time=time))
                del notes_on[i]
                i -= 1
                last_event = t
            i += 1
        for note in range(data.shape[1]):
            if data[t][note] != 0:
                time = int((t - last_event) * atomic)
                messages.append(mido.Message(type='note_on', note=note, velocity=50, time=time))
                notes_on.append((note, t, data[t][note]))
                last_event = t
    while len(notes_on) > 0:
        note = notes_on[0]
        time = int((data.shape[0] - last_event) * atomic)
        messages.append(mido.Message(type='note_off', note=note[0], velocity=0, time=time))
        del notes_on[0]
        last_event = data.shape[0]
    return messages


def saveMidiFile(midifile, messages, meta, filename='output.mid'):
    """
    Updates and saves midi file into output directory.
    :param midifile: initial midi file (MidiFile).
    :param messages: updated music (list of mido.Message objects).
    :param meta: meta information from initial MidiFile.
    :param filename: filename for newly created file.
    """
    tracks = [*meta[:-1], *messages, meta[-1]]
    midifile.tracks = [mido.MidiTrack(tracks)]
    fn = f'output/{filename}'
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    midifile.save(fn)


def main(filename):
    """
    Main function
    """
    mid = mido.MidiFile(filename)
    messages = []
    meta = []
    tempo = 0
    for msg in mid:
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            break

    min_time = tempo
    for msg in mid:
        if msg.type in ['note_on', 'note_off']:
            t = int(mido.second2tick(msg.time, mid.ticks_per_beat, tempo))
            messages.append(msg.copy(time=t))
            if 0 < t < min_time:
                min_time = t
        else:
            meta.append(msg)
    length = int(mido.second2tick(mid.length, mid.ticks_per_beat, tempo))
    temp = (length / min_time) / math.ceil(length / (2 * mid.ticks_per_beat))
    accompaniment_length = math.ceil(temp)
    message_data = messages2data(messages, min_time, length)
    evolved_data = evolution(message_data, accompaniment_length, add_sound=temp < accompaniment_length)
    printNotes(evolved_data)
    messages = data2messages(evolved_data, min_time)
    saveMidiFile(mid, messages, meta, filename)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Error: you should provide the name of the MIDI file.')
        print('Usage: `python3 AndreyKuzmickiy.py AndreyKuzmickiyOutput1.mid`')
        sys.exit(1)
    midfile = sys.argv[1]
    if midfile.split('.')[-1] != 'mid':
        print('Error: you should provide MIDI file (file.mid)')
        sys.exit(1)
    main(midfile)
