import { assertEquals, assertExists, assertRejects, assertSpyCall, assertSpyCalls } from '@std/testing/asserts.ts';
import { describe, it, beforeEach, afterEach } from '@std/testing/bdd.ts';
import { spy, stub } from '@std/testing/mock.ts';
import { AgentCommand, AgentManager, AgentType } from '../../../../src/cli/commands/agent.ts';
import { Logger } from '../../../../src/core/logger.ts';
import { Config } from '../../../../src/core/config.ts';
import { EventBus } from '../../../../src/core/event-bus.ts';

describe('Agent Command Tests', () => {
  let agentCommand: AgentCommand;
  let mockAgentManager: AgentManager;
  let mockLogger: Logger;
  let mockConfig: Config;
  let mockEventBus: EventBus;

  beforeEach(() => {
    mockLogger = {
      info: spy(),
      error: spy(),
      warn: spy(),
      debug: spy(),
    } as unknown as Logger;

    mockConfig = {
      get: stub().returns('test-value'),
      set: spy(),
      getAgentConfig: stub().returns({
        maxAgents: 10,
        defaultType: 'researcher',
        timeout: 30000,
      }),
    } as unknown as Config;

    mockEventBus = {
      emit: spy(),
      on: spy(),
      off: spy(),
    } as unknown as EventBus;

    mockAgentManager = {
      spawn: stub().resolves({ id: 'agent-123', type: 'researcher', name: 'Test Agent' }),
      list: stub().resolves([
        { id: 'agent-1', type: 'researcher', name: 'Agent 1', status: 'active' },
        { id: 'agent-2', type: 'coder', name: 'Agent 2', status: 'idle' },
      ]),
      terminate: stub().resolves(true),
      getStatus: stub().resolves({ status: 'active', tasks: 2, uptime: 3600 }),
    } as unknown as AgentManager;

    agentCommand = new AgentCommand(mockAgentManager, mockLogger, mockConfig, mockEventBus);
  });

  afterEach(() => {
    // Clean up any resources
  });

  describe('spawn subcommand', () => {
    it('should spawn a new agent with default type', async () => {
      await agentCommand.execute(['spawn']);
      
      assertSpyCall(mockAgentManager.spawn, 0, {
        args: ['researcher', undefined],
      });
      assertSpyCall(mockLogger.info, 0, {
        args: ['Agent spawned successfully: agent-123 (researcher)'],
      });
    });

    it('should spawn a new agent with specified type', async () => {
      await agentCommand.execute(['spawn', 'coder']);
      
      assertSpyCall(mockAgentManager.spawn, 0, {
        args: ['coder', undefined],
      });
    });

    it('should spawn a new agent with custom name', async () => {
      await agentCommand.execute(['spawn', 'analyst', '--name', 'DataAnalyzer']);
      
      assertSpyCall(mockAgentManager.spawn, 0, {
        args: ['analyst', 'DataAnalyzer'],
      });
    });

    it('should handle invalid agent type', async () => {
      await assertRejects(
        async () => await agentCommand.execute(['spawn', 'invalid-type']),
        Error,
        'Invalid agent type: invalid-type'
      );
    });

    it('should handle spawn failure', async () => {
      mockAgentManager.spawn = stub().rejects(new Error('Spawn failed'));
      
      await assertRejects(
        async () => await agentCommand.execute(['spawn']),
        Error,
        'Spawn failed'
      );
      
      assertSpyCall(mockLogger.error, 0, {
        args: ['Failed to spawn agent: Spawn failed'],
      });
    });

    it('should emit agent.spawned event', async () => {
      await agentCommand.execute(['spawn', 'researcher']);
      
      assertSpyCall(mockEventBus.emit, 0, {
        args: ['agent.spawned', { id: 'agent-123', type: 'researcher', name: 'Test Agent' }],
      });
    });

    it('should respect max agents limit', async () => {
      mockAgentManager.list = stub().resolves(new Array(10).fill({ id: 'agent', status: 'active' }));
      
      await assertRejects(
        async () => await agentCommand.execute(['spawn']),
        Error,
        'Maximum number of agents (10) reached'
      );
    });
  });

  describe('list subcommand', () => {
    it('should list all agents', async () => {
      await agentCommand.execute(['list']);
      
      assertSpyCall(mockAgentManager.list, 0);
      assertSpyCalls(mockLogger.info, 3); // Header + 2 agents
    });

    it('should handle empty agent list', async () => {
      mockAgentManager.list = stub().resolves([]);
      
      await agentCommand.execute(['list']);
      
      assertSpyCall(mockLogger.info, 0, {
        args: ['No agents currently running'],
      });
    });

    it('should filter agents by type', async () => {
      await agentCommand.execute(['list', '--type', 'researcher']);
      
      assertSpyCall(mockAgentManager.list, 0, {
        args: [{ type: 'researcher' }],
      });
    });

    it('should filter agents by status', async () => {
      await agentCommand.execute(['list', '--status', 'active']);
      
      assertSpyCall(mockAgentManager.list, 0, {
        args: [{ status: 'active' }],
      });
    });

    it('should support JSON output format', async () => {
      const consoleSpy = spy(console, 'log');
      
      await agentCommand.execute(['list', '--format', 'json']);
      
      assertSpyCall(consoleSpy, 0);
      const output = consoleSpy.calls[0].args[0];
      assertEquals(JSON.parse(output).length, 2);
      
      consoleSpy.restore();
    });
  });

  describe('terminate subcommand', () => {
    it('should terminate an agent by ID', async () => {
      await agentCommand.execute(['terminate', 'agent-123']);
      
      assertSpyCall(mockAgentManager.terminate, 0, {
        args: ['agent-123'],
      });
      assertSpyCall(mockLogger.info, 0, {
        args: ['Agent agent-123 terminated successfully'],
      });
    });

    it('should handle terminate failure', async () => {
      mockAgentManager.terminate = stub().resolves(false);
      
      await assertRejects(
        async () => await agentCommand.execute(['terminate', 'agent-123']),
        Error,
        'Failed to terminate agent agent-123'
      );
    });

    it('should terminate all agents with --all flag', async () => {
      await agentCommand.execute(['terminate', '--all']);
      
      assertSpyCalls(mockAgentManager.terminate, 2); // Called for each agent
      assertSpyCall(mockLogger.info, 2, {
        args: ['All agents terminated successfully'],
      });
    });

    it('should emit agent.terminated event', async () => {
      await agentCommand.execute(['terminate', 'agent-123']);
      
      assertSpyCall(mockEventBus.emit, 0, {
        args: ['agent.terminated', { id: 'agent-123' }],
      });
    });
  });

  describe('status subcommand', () => {
    it('should show agent status', async () => {
      await agentCommand.execute(['status', 'agent-123']);
      
      assertSpyCall(mockAgentManager.getStatus, 0, {
        args: ['agent-123'],
      });
      assertSpyCalls(mockLogger.info, 3); // Status details
    });

    it('should handle status retrieval failure', async () => {
      mockAgentManager.getStatus = stub().rejects(new Error('Agent not found'));
      
      await assertRejects(
        async () => await agentCommand.execute(['status', 'agent-123']),
        Error,
        'Agent not found'
      );
    });

    it('should show detailed status with --detailed flag', async () => {
      mockAgentManager.getStatus = stub().resolves({
        status: 'active',
        tasks: 2,
        uptime: 3600,
        memory: { used: 100, total: 500 },
        cpu: 25.5,
        lastTask: { name: 'Research task', completedAt: new Date() },
      });
      
      await agentCommand.execute(['status', 'agent-123', '--detailed']);
      
      assertSpyCalls(mockLogger.info, 6); // More details shown
    });
  });

  describe('error handling', () => {
    it('should handle unknown subcommand', async () => {
      await assertRejects(
        async () => await agentCommand.execute(['unknown']),
        Error,
        'Unknown agent subcommand: unknown'
      );
    });

    it('should handle missing required arguments', async () => {
      await assertRejects(
        async () => await agentCommand.execute(['terminate']),
        Error,
        'Agent ID required'
      );
    });

    it('should validate agent types', async () => {
      const validTypes: AgentType[] = ['researcher', 'coder', 'analyst', 'tester', 'architect'];
      
      for (const type of validTypes) {
        await agentCommand.execute(['spawn', type]);
        assertSpyCall(mockAgentManager.spawn, 0, {
          args: [type, undefined],
        });
        mockAgentManager.spawn.restore();
        mockAgentManager.spawn = stub().resolves({ id: 'agent-123', type, name: 'Test Agent' });
      }
    });
  });

  describe('concurrency and race conditions', () => {
    it('should handle concurrent spawn requests', async () => {
      const promises = Array(5).fill(null).map((_, i) => 
        agentCommand.execute(['spawn', 'researcher', '--name', `Agent-${i}`])
      );
      
      await Promise.all(promises);
      
      assertSpyCalls(mockAgentManager.spawn, 5);
    });

    it('should prevent duplicate agent names', async () => {
      mockAgentManager.spawn = stub()
        .onFirstCall().resolves({ id: 'agent-1', type: 'researcher', name: 'Duplicate' })
        .onSecondCall().rejects(new Error('Agent name already exists'));
      
      await agentCommand.execute(['spawn', 'researcher', '--name', 'Duplicate']);
      
      await assertRejects(
        async () => await agentCommand.execute(['spawn', 'researcher', '--name', 'Duplicate']),
        Error,
        'Agent name already exists'
      );
    });
  });

  describe('integration with other components', () => {
    it('should update config after spawning agent', async () => {
      await agentCommand.execute(['spawn', 'researcher']);
      
      assertSpyCall(mockConfig.set, 0, {
        args: ['lastSpawnedAgent', 'agent-123'],
      });
    });

    it('should log all operations', async () => {
      await agentCommand.execute(['spawn', 'researcher']);
      await agentCommand.execute(['list']);
      await agentCommand.execute(['status', 'agent-123']);
      await agentCommand.execute(['terminate', 'agent-123']);
      
      assertSpyCalls(mockLogger.info, 8); // All operations logged
      assertSpyCalls(mockLogger.debug, 4); // Debug logs for each operation
    });
  });
});