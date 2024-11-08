import json
import os
import csv
import re
from itertools import groupby
from operator import itemgetter
from typing import Union, Optional
import urllib.parse

import unicodedata


def read_markdown_file(file_path: str) -> list[str]:
    """Read a markdown file and return its lines, removing the first two if the first line is '# Untitled'."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()  # Read all lines from the file

        return [line.rstrip() for line in lines]  # Return lines, stripping newline characters

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


def remove_color_prefix(text: str) -> str:
    """Remove any leading non-printable or invisible characters from the string."""
    # Use regex to match leading non-visible characters
    cleaned_text = re.sub(r'^[\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]+', '', text)
    return cleaned_text


def load_csv_as_table(csv_file_path) -> list[dict[str, str]]:
    """Load a CSV file, ignore the header, and return data as a list of lists."""
    try:
        data = []
        columns = None

        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            columns = [remove_color_prefix(column) for column in next(reader)]

            data = [
                {columns[i]: row[i] for i in range(len(columns))}
                for row in reader
            ]

        return data

    except Exception as e:
        print(f"Error loading CSV file {csv_file_path}: {e}")
        return []


def get_record_type(record: dict[str, str]) -> str:
    general = {*('Title', 'Status', 'Type', 'Assignee')}
    concept = {*('Question', 'Status')}
    question = {*('Type', 'Content', 'Scenario')}
    scenario = {*()}

    if general == record.keys():
        return 'General'

    if concept == record.keys():
        return 'Concept'

    if question == record.keys():
        return 'Question'

    return 'Scenario'


def get_table_type(records: list[dict[str, str]]) -> Optional[str]:
    if len(records) == 0:
        return None

    return get_record_type(records[0])


def get_id(record: dict[str, str]) -> Optional[str]:
    record_type = get_record_type(record)

    title = None

    if record_type == 'General':
        title = record['Title']

    if record_type == 'Concept':
        title = record['Question']

    if record_type == 'Question':
        title = record['Content']

    if title is not None:
        title = title.replace('\n', '').strip()

    return title


def extract_title_from_markdown(lines: list[str]) -> Optional[str]:
    if not lines[0].startswith('#'):
        return None

    title = ''

    for line in lines:
        if line == '':
            break

        title = title + ' ' + line

    return title[2:].strip()


def extract_columns_from_markdown(lines: list[str]) -> dict:
    """Extract the first lines from markdown and return values and modified markdown."""
    if not lines[0].startswith('#'):
        return {}

    title_lines = []
    for line in lines:
        if line == '':
            break

        title_lines.append(line)

    lines = lines[len(title_lines):]

    blank_lines = 0
    for line in lines:
        if line != '':
            break
        blank_lines += 1
    if blank_lines != 0:
        lines = lines[blank_lines:]

    properties = {}
    for line in lines:
        if not ':' in line:
            break

        # Directly append the value from each line
        parts = line.split(':', 1)
        key = parts[0].strip()
        value = parts[1].strip()
        properties[key] = value

    lines = lines[len(properties):]

    blank_lines = 0
    for line in lines:
        if line != '':
            break
        blank_lines += 1
    if blank_lines != 0:
        lines = lines[blank_lines:]

    # Remaining lines after the blank line
    new_markdown_lines = lines

    # Return the CSV values as a single row and the modified markdown
    return {'properties': properties, 'content': new_markdown_lines}


def get_csv_filenames(folder_path: str) -> list[str]:
    """Return a list of all CSV filenames in the specified folder."""
    csv_files = []

    try:
        # List all items in the folder
        files = os.listdir(folder_path)

        # Filter for CSV files
        csv_files = [file for file in files if file.endswith('.csv')]

    except Exception as e:
        print(f"Error accessing folder {folder_path}: {e}")

    return csv_files


def get_markdown_filenames(folder_path: str) -> set[str]:
    """Return a set of all markdown filenames in the specified folder."""
    markdown_files = []

    try:
        # List all files in the folder
        files = os.listdir(folder_path)

        # Filter for markdown files
        markdown_files = {file for file in files if file.endswith('.md')}

    except Exception as e:
        print(f"Error accessing folder {folder_path}: {e}")

    return markdown_files


def extract_csv_filename(notion_string: str) -> Union[str, None]:
    """Extract the CSV filename from a Notion-style link."""
    # Updated regex pattern to capture the filename directly
    pattern = r'\(([^)]+\.csv)\)'  # Match the content inside parentheses that ends with .csv

    match = re.search(pattern, notion_string)
    if match:
        return match.group(1)  # Return only the captured filename
    else:
        return None  # Return None if no match is found


def is_notion_table_reference(text: str) -> bool:
    """Determine if a string is a Notion table reference."""
    # Updated regex pattern to allow for encoded characters and varied formats
    notion_table_pattern = r'^\[.+?\]\([a-zA-Z0-9%\-._]{10,}\)$'

    return bool(re.match(notion_table_pattern, text))


def decode_str(to_decode: str) -> str:
    return urllib.parse.unquote(to_decode)


def sanitize_name(name: str) -> str:
    """Remove illegal characters from a Windows filename or folder name and truncate it to 50 characters."""
    # Define a regex pattern for illegal characters
    name = name.replace('/', ' ')
    illegal_chars_pattern = r'[<>:"/\\|?*\.]'

    # Remove illegal characters
    sanitized = re.sub(illegal_chars_pattern, '', name)

    # Split the name into base and extension (if applicable)
    if name.endswith('.md') or name.endswith('.csv'):
        base_name, ext = os.path.splitext(sanitized)
    else:
        base_name, ext = sanitized, ''

    # Truncate the base name to a maximum of 50 characters
    if len(base_name) > 50:
        base_name = base_name[:50]

    base_name = base_name.strip()
    base_name = unicodedata.normalize('NFC', base_name)

    # Return the sanitized and truncated name, including the extension
    return base_name + ext


def find_markdown_file_by_csv(csv_filename: str, markdown_files: dict[str, list[str]]) -> Union[str, None]:
    """Find the markdown file that references the specified CSV filename."""
    csv_filename = unicodedata.normalize('NFC', csv_filename)

    for markdown_file, lines in markdown_files.items():
        for line in lines:
            if is_notion_table_reference(line):
                extracted_csv_filename = extract_csv_filename(line)
                extracted_csv_filename = decode_str(extracted_csv_filename)
                extracted_csv_filename = sanitize_name(extracted_csv_filename)

                if extracted_csv_filename == csv_filename:
                    return markdown_file

    return None


def strip_notion_id(filename: str) -> str:
    """Strip the Notion ID from a filename."""
    return filename[:-36]


def find_csv_file_by_markdown(markdown_filename: str, csv_files: dict[str, list[dict[str, str]]]) -> Union[str, None]:
    """Find the CSV file that is referenced by the specified markdown filename."""
    markdown_filename = unicodedata.normalize('NFC', markdown_filename)

    for csv_filename, records in csv_files.items():
        for record in records:
            print(record)
            record_id = get_id(record)
            print(record_id)
            record_id = unicodedata.normalize('NFC', record_id)

            if record_id == markdown_filename:
                return csv_filename

    return None


def get_root_markdown_file(markdown_files: dict[str, dict[str]], csv_files: dict[str, list[dict[str, str]]]) -> str:
    unreferenced_markdown_files = [
        markdown_file
        for markdown_file in markdown_files.keys()
        if find_csv_file_by_markdown(markdown_file, csv_files) is None
    ]
    assert len(unreferenced_markdown_files) == 1

    return unreferenced_markdown_files[0]


def inline_file(csv_files, markdown_files, filename) -> list[str]:
    print(f'Processing {filename}')

    lines = [f'# {filename}']

    if filename in markdown_files:
        for line in markdown_files[filename]['content']:
            print(f'markdown line {line}')
            if is_notion_table_reference(line):
                line = extract_csv_filename(line)
                line = decode_str(line)
                print(f'table line {line}')
                if line in csv_files:
                    table = csv_files[line]
                    for row in table:
                        row_id = get_id(row)
                        row_id = unicodedata.normalize('NFC', row_id)
                        print(f'row id {row_id}')
                        if len(row_id) > 0:
                            lines.append('#{')
                            lines.extend(inline_file(csv_files, markdown_files, row_id))
                            lines.append('#}')
            else:
                lines.append(line)

    lines.append('')

    return lines


def check_for_dupes(markdown_files: list[dict[str, str]]):
    markdown_files = sorted(markdown_files, key=itemgetter("filename"))
    markdown_files = {
        filename: list(group) for filename, group in groupby(markdown_files, key=itemgetter("filename"))
    }
    print({name: content for name, content in markdown_files.items() if len(content) > 1})
    assert len({name: content for name, content in markdown_files.items() if len(content) > 1}) == 0


def extract_markdown(folder: str) -> str:
    csv_files = get_csv_filenames(folder)
    csv_files = [csv_file for csv_file in csv_files if not csv_file.endswith('_all.csv')]
    csv_files = {csv_file: load_csv_as_table(os.path.join(folder, csv_file)) for csv_file in csv_files}

    markdown_files = get_markdown_filenames(folder)
    markdown_files = {markdown_file: read_markdown_file(os.path.join(folder, markdown_file)) for markdown_file in markdown_files}
    markdown_files = [{'filename': extract_title_from_markdown(markdown_lines), **extract_columns_from_markdown(markdown_lines)} for
                      markdown_file, markdown_lines in markdown_files.items()]
    markdown_files = [markdown_file for markdown_file in markdown_files if markdown_file['content'] and markdown_file['filename'] not in {*('Untitled', 'Question', 'Scenario', 'Concept')}]
    check_for_dupes(markdown_files)
    markdown_files = {markdown_file['filename']: markdown_file for markdown_file in markdown_files}

    root_markdown_file = get_root_markdown_file(markdown_files, csv_files)
    inlined = inline_file(csv_files, markdown_files, root_markdown_file)

    return inlined

if __name__ == "__main__":
    root_folder = "C:\\Users\\xteam\\Downloads\\18e2d61c-3746-4654-a0e2-fa65834e325b_Export-41e8fb2a-1bb1-4ebc-bfac-cef35eb1358a"
    output_file = "../consolidated.md"

    markdown = extract_markdown(root_folder)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown))
