"""
This module hosts routines I couldn't find a place for elsewhere. 
"""

from typing import Union
import pyarrow as pa

from .primitives import *


def arrow_type_to_py(arrow_typ) -> Union[LITERAL_TYPES, None]:
    typ = None
    if pa.types.is_integer(arrow_typ):
        typ = int
    elif pa.types.is_floating(arrow_typ):
        typ = float
    elif pa.types.is_boolean(arrow_typ):
        typ = bool
    elif pa.types.is_string(arrow_typ):
        typ = str
    return typ  # type: ignore


def get_types_from_dataset(dataset):
    """Read the schema of an Arrow dataset to determine the types of the columns."""
    seq_type_map = dict()
    events_type_map = dict()

    schema = dataset.schema
    for name in schema.names:
        ftype = schema.field(name).type
        if name != "events":
            typ = arrow_type_to_py(ftype)
            if typ is None:
                print(f"field type unclassified: {name}")
            else:
                seq_type_map[name] = typ
        else:
            num_fields = ftype.value_type.num_fields
            for i in range(num_fields):
                field = ftype.value_type[i]
                typ = arrow_type_to_py(field.type)
                if typ is None:
                    print(f"event field type unclassified: {field.name}")
                else:
                    events_type_map[field.name] = typ

    return seq_type_map, events_type_map
