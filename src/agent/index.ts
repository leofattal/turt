/**
 * Coding-agent layer (see PRD "Coding Agent").
 *
 * Planned capabilities: read repositories, plan tasks, refactor safely,
 * generate tests, explain/review code, benchmark, detect security issues,
 * produce documentation. The agent composes the model (src/nn + src/infer),
 * long-term memory (src/memory), and the tool registry (src/tools).
 */

import type { ToolRegistry } from "../tools/tool.js";

export interface AgentStep {
  thought: string;
  tool?: { name: string; input: unknown };
  result?: unknown;
}

export interface Agent {
  readonly tools: ToolRegistry;
  run(task: string): Promise<AgentStep[]>;
}
