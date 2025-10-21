import json

def get_schema(data, level=0):
    """Recursively infer schema of JSON data."""
    if isinstance(data, dict):
        schema = {}
        for k, v in data.items():
            schema[k] = get_schema(v, level + 1)
        return schema
    elif isinstance(data, list):
        if len(data) > 0:
            # Only inspect first element to avoid large memory usage
            return [get_schema(data[0], level + 1)]
        else:
            return []
    else:
        return type(data).__name__


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python json_schema_viewer.py <path_to_json>")
        return

    file_path = sys.argv[1]
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        #print(len(data['images']))
       # print(data[:2])
        print(data['annotations'][:2])

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    schema = get_schema(data)
    print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    main()
