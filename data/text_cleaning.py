# -*- coding: utf-8 -*-
"""This module provides some functions for cleaning raw text."""

import re
import html
import string

def clean_string(text):
    """
    Removes html tags and classification codes from a string of text.

    Parameters
    ----------
    text : str
        A string of text.

    Returns
    -------
    str
        Cleaned string of text.

    """
    text = html.unescape(text)
    text = text.strip(' "\'.')

    # general
    text = re.sub('\\t', '', text)
    text = re.sub('\\n', '', text)
    text = re.sub('>\s+', '>', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub('<head>.*<\s*(\/head|body)[^>]*>', '', text, flags=re.I) # remove <head> to </head>
    text = re.sub('<a\s+href="([^"]+)"[^>]*>.*</a>', '', text, flags=re.I) # remove URLs
    text = re.sub('[ \t]*<[^<]*?/?>', '', text) # remove remaining tags
    text = re.sub('^[^A-Za-z]+$', '', text)
    text = re.sub('\"', '', text)

    # remove AMS and JEL codes from text
    text = re.sub('ams classification(s|(\snumbers))*(:|;)\s([0-9]{2}[-a-z][0-9]{2}(,|;)*\s*)+', '', text, flags=re.I)
    text = re.sub('jel classification(s|(\snumbers))*(:|;)\s([-a-rz][0-9]{1,2}(,|;)*\s*)+', '', text, flags=re.I)
    text = re.sub('jel codes*(:|;)\s([-a-rz][0-9]{1,2}(,|;)*\s*)+', '', text, flags=re.I)
    text = re.sub('^([a-rz][0-9]{1,2}.\s*)+', '', text, flags=re.I)
    text = re.sub('([a-rz][0-9]{1,2}.\s*)+$', '', text, flags=re.I)

    # remove remaining numbers from text
    text = re.sub(r'\d+', '', text)

    return text.strip()

def clean_abstract(text, fasttext_model):
    """
    Cleans abstracts of papers.

    This function cleans abstracts by removing specific phrases not related to
    the content of the paper and retains the English portion of multilingual
    abstracts. Note that abstract cleaning is done using a fixed list of phrases
    and regular expressions and should not be considered an exhaustive list,
    thus caution is advised.

    Parameters
    ----------
    text : str
        Raw abstract.
    fasttext_model: FastText object
        FastText model for determining the language of a string of text.

    Returns
    -------
    str
        Cleaned abstract.

    """
    text = clean_string(text)

    # remove specific phrases
    text = re.sub('\(*no abstract, this is a discusion paper\)*', '', text, flags=re.I)
    text = re.sub('\(*see PDF\)*', '', text, flags=re.I)
    text = re.sub('\(*see .+ for abstract\)*', '', text, flags=re.I)
    text = re.sub('\[*Abstract in English is missing\]*', '', text, flags=re.I)
    text = re.sub('\[*abstract missing - contribution appeared in the programme\]*', '', text, flags=re.I)
    text = re.sub('\[*abstract missing - presentation attached\]*', '', text, flags=re.I)
    text = re.sub('please refer to full text', '', text, flags=re.I)
    text = re.sub('no abstract received', '', text, flags=re.I)

    invalid_abstracts = ['abstract', 'book review', 'coming soon', 'contents', 'delete', 'eres:conference', 'foreword', 'nobel prize lecture', 'none', 'na', 'n.a',  'no abstract', 'not available', 'tb', 'tba']
    if text.strip(' .').lower() in invalid_abstracts:
        return ''

    # split delimited bilingual text where possible
    m = re.search(r'(?<=\(english\)).*?(?=\(fran.ais\))', text, flags=re.I)
    if m:
        return m.group(0).strip(' _*')

    m = re.search(r'(?<=\(fran.ais\)).*?(?=\(english\)).*?', text, flags=re.I)
    if m:
        return (re.split('(english)', text, 1, flags=re.I)[1]).strip(' _*')

    m = re.search(r'(?<=\(VA\)).*?(?=\(VF\))', text, flags=re.I)
    if m:
        return m.group(0).strip(' _*')

    m = re.search(r'(?<=\(VF\)).*?(?=\(VA\)).*?', text, flags=re.I)
    if m:
        return (re.split('(VA)', text, 1, flags=re.I)[1]).strip(' _*')

    m = re.search(r'(?<=english abstract).*?(?=french abstract)', text, flags=re.I)
    if m:
        return m.group(0).strip(' _*')

    m = re.search(r'(?<=english abstract).*?(?=resumen)', text, flags=re.I)
    if m:
        return m.group(0).strip(' _*-')

    m = re.search(r'(?<=abstract).*?(?=resumen)', text, flags=re.I)
    if m:
        return m.group(0).strip(' _*-')

    m = re.search(r'(?<=resumen).*?(?=abstract)', text, flags=re.I)
    if m:
        return (re.split('abstract', text, 1, flags=re.I)[1]).strip(' _*-')

    m = re.search(r'(?<=spanish abstract).*?(?=english abstract)', text, flags=re.I)
    if m:
        return (re.split('english abstract', text, 1, flags=re.I)[1]).strip(' _*-')

    m = re.search(r'(?<=bulgarian abstract).*?(?=english abstract)', text, flags=re.I)
    if m:
        return (re.split('english abstract', text, 1, flags=re.I)[1]).strip(' _*-')

    m = re.search(r'(?<=bulgarian).*?(?=abstract)', text, flags=re.I)
    if m:
        return (re.split('abstract', text, 1, flags=re.I)[1]).strip(' _*-')

    m = re.search(r'(?<=romanian abstract).*?(?=english abstract)', text, flags=re.I)
    if m:
        return (re.split('english abstract', text, 1, flags=re.I)[1]).strip(' _*-')

    m = re.search(r'(?<=polish abstract).*?(?=english abstract)', text, flags=re.I)
    if m:
        return (re.split('english abstract', text, 1, flags=re.I)[1]).strip(' _*-')

    m = re.search(r'(?<=russian abstract).*?(?=english abstract)', text, flags=re.I)
    if m:
        return (re.split('english abstract', text, 1, flags=re.I)[1]).strip(' _*-')

    m = re.search(r'(?<=english abstract).*?(?=serbian abstract)', text, flags=re.I)
    if m:
        return m.group(0).strip(' _*-')

    m = re.search(r'(?<=serbian abstract).*?(?=english abstract)', text, flags=re.I)
    if m:
        return (re.split('english abstract', text, 1, flags=re.I)[1]).strip(' _*-')

    m = re.search(r'(?<=(?:serbian|russian)).*?(?=english)', text, flags=re.I)
    if m:
        return (re.split('english', text, 1, flags=re.I)[1]).strip(' _*-')

    m = re.search(r'(?<=(?:english)).*?(?=serbian)', text, flags=re.I)
    if m:
        return m.group(0).strip(' _*-')

    m = re.search(r'(?<=\[en\]).*?(?=\[it\])', text, flags=re.I)
    if m:
        return m.group(0).strip(' _*-')

    m = re.search(r'(?<=italian abstract).*?(?=english abstract)', text, flags=re.I)
    if m:
        return (re.split('english abstract', text, 1, flags=re.I)[1]).strip(' _*-')

    m = re.search(r'(?<=\[in macedonian\]).*?(?=\[in english\])', text, flags=re.I)
    if m:
        return (re.split('\[in english\]', text, 1, flags=re.I)[1]).strip(' _*-')

    m = re.search(r'(?<=in macedonian).*?(?=in english)', text, flags=re.I)
    if m:
        return (re.split('in english', text, 1, flags=re.I)[1]).strip(' _*-')

    m = re.search(r'(?<=abstrak).*?(?=abstract)', text, flags=re.I)
    if m:
        return (re.split('abstract', text, 1, flags=re.I)[1]).strip(' _*-')

    # Other delimiters
    m = re.split('(?:\*{2,}|_{2,}|-{2,})', text)
    if len(m) > 1:
        for i in m:
            if len(i) > 0:
                try:
                    # Determine language of split text
                    lang = fasttext_model.predict(i)[0][0][-2:]
                except Exception as e:
                    print(e)
                    print('TEXT: %s' % i)
                else:
                    if lang == 'en':
                        return i.strip(' _*-')

    return text.strip()

def extract_english(text, fasttext_model):
    """
    Extracts English sentences from a multi-sentence string.

    Parameters
    ----------
    text : str
        Multi-sentence string.
    fasttext_model : FastText object
        FastText model for determining the language of a string of text.

    Returns
    -------
    str
        String consisting of English sentences.

    """
    sentences = text.split('. ')
    english_text = ''
    for sentence in sentences:
        try:
            lang = fasttext_model.predict(sentence)[0][0][-2:]
        except Exception as e:
            print(e)
            print('TEXT: %s' % text)
        else:
            if lang == 'en':
                english_text = english_text + ' ' + sentence.strip() + '.'

    return english_text.strip()

def clean_jel(text, labels):
    """
    Retrieves JEL codes from a string.

    This function takes a string of text and removes words, AMS codes,
    digits not part of the JEL codes, and punctuation marks, such that only
    the JEL codes are left, then converts the string into a set of JEL codes.

    Parameters
    ----------
    text : str
        raw string containing JEL codes.
    labels: set of str
        set containing all existing JEL codes.

    Returns
    -------
    set of str
        set of JEL codes present in input string.

    """
    text = html.unescape(text)

    text = re.sub('JEL|AMS|classification|codes|code|system', '', text, flags=re.I)
    text = re.sub('^([A-Za-z]\s*)+$', '', text) # Remove text that contains only letters
    text = re.sub('^([0-9]\s*)+$', '', text) # Remove text that contains only numbers
    text = re.sub('([0-9]{2}[-A-Za-z][0-9]{2})', '', text) # Remove AMS classification codes
    text = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', text) # Replace punctuation by spaces
    text = re.sub('\s+', ' ', text) # Remove excess spaces

    jel_codes = re.findall('(?<![A-Z])[A-RZ][0-9]{2}(?!\S)', text.strip().upper())
    valid_labels = [i for i in jel_codes if i in labels]
    return set(valid_labels)

def remove_symbols(text, keep=string.printable):
    """
    Returns filtered string.

    Parameters
    ----------
    text : str
        string of text.
    keep : str or set or list, optional
        characters that should be kept in the string. The default is string.printable.

    Returns
    -------
    str
        filtered string of text.

    """
    text = ''.join(filter(lambda x: x in keep, text))
    return text.strip()



