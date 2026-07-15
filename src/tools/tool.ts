/**
 * Extensible tool interface (see PRD "Tool Use").
 *
 * Concrete tools — file operations, code editing, terminal commands, git,
 * documentation lookup, API requests, search — each implement `Tool` and
 * register with a `ToolRegistry`. The agent layer (src/agent) selects and
 * invokes tools by name with JSON-serializable inputs.
 */

export interface Tool<Input = unknown, Output = unknown> {
  readonly name: string;
  readonly description: string;
  run(input: Input): Promise<Output>;
}

export class ToolRegistry {
  private tools = new Map<string, Tool>();

  register(tool: Tool): void {
    if (this.tools.has(tool.name)) throw new Error(`Tool "${tool.name}" is already registered`);
    this.tools.set(tool.name, tool);
  }

  get(name: string): Tool | undefined {
    return this.tools.get(name);
  }

  list(): Array<{ name: string; description: string }> {
    return [...this.tools.values()].map(({ name, description }) => ({ name, description }));
  }

  async invoke(name: string, input: unknown): Promise<unknown> {
    const tool = this.tools.get(name);
    if (!tool) throw new Error(`Unknown tool "${name}"`);
    return tool.run(input);
  }
}
