import argparse
import re
from copy import deepcopy
from lxml import etree
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import AutoTokenizer, M2M100ForConditionalGeneration


@dataclass
class TranslatorConfig:
    """
    Configuration class for subtitle translation tasks.

    Attributes:
        model (M2M100ForConditionalGeneration): A pre-trained M2M100 model from
            Hugging Face's transformers library for machine translation.
        tokenizer (AutoTokenizer): Corresponds to the M2M100 model and is used
            for converting input text into a format suitable for the model.
        batch_size (int): Number of subtitle entries to process in a batch for
            efficient translation, reducing model calls. Default is 10.

    Example:
        >>> model = M2M100ForConditionalGeneration.from_pretrained(
        ...     'facebook/m2m100_418M')
        >>> tokenizer = AutoTokenizer.from_pretrained('facebook/m2m100_418M')
        >>> config = TranslatorConfig(model=model, tokenizer=tokenizer,
        ...                           batch_size=10)
    """

    model: M2M100ForConditionalGeneration
    tokenizer: AutoTokenizer
    batch_size: int = 10


def translate_batch(text_batch, dest_lang, model, tokenizer):
    """
    Translates a batch of text into the specified destination language.

    This function takes a list of text strings and the target language code, and
    uses the provided model and tokenizer to generate the translations. It handles
    the tokenization of the input text, the generation of translated tokens, and
    the decoding of these tokens back into translated strings.

    Args:
        text_batch (list of str): A list of text strings to be translated.
        dest_lang (str): The target language code to which the text should be
            translated.
        model (transformers.PreTrainedModel): A pre-trained model from the
            `transformers` library.
        tokenizer (transformers.PreTrainedTokenizer): A tokenizer that corresponds
            to the `model`.

    Returns:
        list of str: A list of translated text strings.

    Example:
        >>> model_name = 'facebook/m2m100_418M'
        >>> model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        >>> tokenizer = AutoTokenizer.from_pretrained(model_name)
        >>> text_batch = ["Hello, world!", "How are you?"]
        >>> dest_lang = 'fr'
        >>> translate_batch(text_batch, dest_lang, model, tokenizer)
        ['Bonjour, monde !', 'Comment ça va ?']
    """

    model_inputs = tokenizer(
        text_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
    gen_tokens = model.generate(
        **model_inputs, forced_bos_token_id=tokenizer.get_lang_id(dest_lang))
    translations = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    return translations


@dataclass
class GElement:
    id: str
    ctype: str
    text: str


@dataclass
class TextContainer:
    text: Optional[str] = None
    g_elements: List[GElement] = field(default_factory=list)


@dataclass
class TransUnit:
    id: str
    source: TextContainer = field(default_factory=TextContainer)
    target: TextContainer = field(default_factory=TextContainer)


@dataclass
class File:
    original: str
    datatype: str
    source_language: str
    target_language: str
    trans_units: List[TransUnit] = field(default_factory=list)


def parse_xliff(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = etree.parse(file)
    root = tree.getroot()

    ns = {'default': root.nsmap[None]}
    files = []

    for file_elem in root.findall('default:file', namespaces=ns):
        file_original = file_elem.get('original')
        file_datatype = file_elem.get('datatype')
        file_source_language = file_elem.get('source-language')
        file_target_language = file_elem.get('target-language')
        trans_units = []

        for trans_unit_elem in file_elem.findall('.//default:trans-unit', namespaces=ns):
            trans_unit_id = trans_unit_elem.get('id')
            source = parse_text_container(trans_unit_elem.find(
                'default:source', namespaces=ns), ns)
            target = parse_text_container(trans_unit_elem.find(
                'default:target', namespaces=ns), ns)

            trans_units.append(
                TransUnit(id=trans_unit_id, source=source, target=target))

        files.append(File(
            original=file_original,
            datatype=file_datatype,
            source_language=file_source_language,
            target_language=file_target_language,
            trans_units=trans_units
        ))

    return files


def parse_text_container(elem, ns):
    text_container = TextContainer()
    if elem is not None:
        # S'il y a du texte directement dans l'élément, le récupérer
        if elem.text:
            text_container.text = elem.text.strip()

        # Parcourir les éléments <g> s'ils existent
        for g_elem in elem.findall('default:g', namespaces=ns):
            g_id = g_elem.get('id')
            g_ctype = g_elem.get('ctype')
            g_text = ''.join(g_elem.itertext()).strip()
            text_container.g_elements.append(
                GElement(id=g_id, ctype=g_ctype, text=g_text))

    return text_container


def copy_source_to_target(files: List[File]):
    """
    Copies the contents from <source> to <target> for each TransUnit in each File.
    """
    for file in files:
        for trans_unit in file.trans_units:
            # If there is a source, copy its content to the target
            if trans_unit.source:
                if trans_unit.target is None:
                    trans_unit.target = deepcopy(trans_unit.source)
                else:
                    trans_unit.target.text = deepcopy(trans_unit.source.text)
                    trans_unit.target.g_elements = deepcopy(
                        trans_unit.source.g_elements)


def contains_letters(text):
    return bool(re.search('[a-zA-Z]', text))


def translate_targets(files, dest_lang, model, tokenizer):
    # Préparer la liste de textes à traduire
    text_to_translate = []
    mapping = []  # Pour suivre où chaque texte doit aller

    for file in files:
        for trans_unit in file.trans_units:
            # Ajouter le texte source à la liste de traduction si ce n'est pas vide et contient des lettres
            if trans_unit.source.text and trans_unit.source.text.strip() and contains_letters(trans_unit.source.text):
                text_to_translate.append(trans_unit.source.text)
                # None pour les textes qui ne sont pas dans <g>
                mapping.append((trans_unit, 'text', None))
            else:
                # Conserver le texte original si le texte ne contient pas de lettres
                trans_unit.target.text = trans_unit.source.text

            # Faire de même pour les éléments <g> s'ils existent et ne sont pas vides
            for g_element in trans_unit.source.g_elements:
                if g_element.text and g_element.text.strip() and contains_letters(g_element.text):
                    text_to_translate.append(g_element.text)
                    mapping.append((trans_unit, 'g', g_element))
                else:
                    # Trouver l'élément <g> correspondant dans `target` et conserver le texte original
                    corresponding_g_element = next(
                        (ge for ge in trans_unit.target.g_elements if ge.id == g_element.id), None)
                    if corresponding_g_element:
                        corresponding_g_element.text = g_element.text

    # Traduire en une seule fois pour plus d'efficacité, si on a des textes à traduire

    print(text_to_translate)

    if text_to_translate:
        translated_texts = translate_batch(
            text_to_translate, dest_lang, model, tokenizer)

        # Assigner les textes traduits à leur emplacement respectif dans `target`
        for (trans_unit, text_type, g_element), translation in zip(mapping, translated_texts):
            if text_type == 'text':
                # mise à jour du texte directement dans `target`
                trans_unit.target.text = translation
            elif text_type == 'g' and g_element is not None:
                # Trouver l'élément <g> correspondant dans `target` et le mettre à jour
                corresponding_g_element = next(
                    (ge for ge in trans_unit.target.g_elements if ge.id == g_element.id), None)
                if corresponding_g_element:
                    corresponding_g_element.text = translation
    else:
        print("Aucun texte à traduire.")

    return files


def build_xliff(files, target_language):
    # Créer l'élément racine avec les espaces de noms nécessaires
    nsmap = {
        None: "urn:oasis:names:tc:xliff:document:1.2",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xhtml": "http://www.w3.org/1999/xhtml"
    }
    # Créer l'élément racine avec les espaces de noms nécessaires
    xliff_elem = etree.Element("xliff", nsmap=nsmap, version="1.2")
    xliff_elem.set("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation",
                   "urn:oasis:names:tc:xliff:document:1.2 http://docs.oasis-open.org/xliff/v1.2/os/xliff-core-1.2-strict.xsd")

    for file in files:

        file_attributes = {
            'original': file.original if file.original is not None else '',
            'datatype': file.datatype if file.datatype is not None else '',
            'source-language': file.source_language if file.source_language is not None else target_language,
            'target-language': file.target_language if file.target_language is not None else target_language,
        }
        # Créer l'élément <file> pour chaque fichier
        file_elem = etree.SubElement(xliff_elem, "file", file_attributes)

        # Créer et ajouter le <body>
        body_elem = etree.SubElement(file_elem, "body")

        for trans_unit in file.trans_units:
            # Créer l'élément <trans-unit>
            trans_unit_elem = etree.SubElement(
                body_elem, "trans-unit", {"id": trans_unit.id})

            # Ajouter l'élément <source> et son contenu
            source_elem = etree.SubElement(trans_unit_elem, "source")
            if trans_unit.source.text:
                source_elem.text = trans_unit.source.text
            for g_element in trans_unit.source.g_elements:
                g_elem = etree.SubElement(source_elem, "g", {
                    "id": g_element.id,
                    "ctype": g_element.ctype
                })
                g_elem.text = g_element.text

            # Ajouter l'élément <target> et son contenu
            target_elem = etree.SubElement(trans_unit_elem, "target")
            if trans_unit.target.text:
                target_elem.text = trans_unit.target.text
            for g_element in trans_unit.target.g_elements:
                g_elem = etree.SubElement(target_elem, "g", {
                    "id": g_element.id,
                    "ctype": g_element.ctype
                })
                g_elem.text = g_element.text

    # Convertir l'arbre XML en une chaîne de caractères
    return etree.tostring(xliff_elem, pretty_print=True, encoding='UTF-8', xml_declaration=True).decode()


def main():
    """
    TODO docstring
    """

    parser = argparse.ArgumentParser(
        description='Translate XLF/XLIFF files using batch processing.')
    parser.add_argument(
        'source', help='The source subtitle file to translate.')
    parser.add_argument(
        'destination', help='The subtitle file where to save the translation.')
    parser.add_argument(
        'language', help='The target language code for the translation.')
    args = parser.parse_args()

    model_name = "facebook/m2m100_418M"
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    files = parse_xliff(args.source)
    copy_source_to_target(files)
    files2 = translate_targets(files, args.language, model, tokenizer)

    translated_xliff_content = build_xliff(files2, args.language)
    print(translated_xliff_content)

    with open(args.destination, 'w', encoding='utf-8') as f:
        f.write(translated_xliff_content)


if __name__ == "__main__":
    main()
