import attrs
from agentlib import AgentVariable


def make_table_for_agent_variable():
    fields = attrs.fields(AgentVariable)
    lines = ["field_name;default;description"]
    for fld in fields:
        description = fld.metadata.get("description", "").replace("\n", "")
        lines.append(f""" {fld.name} ; {fld.default} ; {description} """)
    csv = "\n".join(lines)
    with open("agentvariable.csv", "w") as f:
        f.write(csv)
    print(csv)


if __name__ == "__main__":
    make_table_for_agent_variable()
