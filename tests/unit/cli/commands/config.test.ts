import { assertEquals, assertExists, assertRejects, assertSpyCall, assertSpyCalls } from '@std/testing/asserts.ts';
import { describe, it, beforeEach, afterEach } from '@std/testing/bdd.ts';
import { spy, stub } from '@std/testing/mock.ts';
import { ConfigCommand } from '../../../../src/cli/commands/config.ts';
import { Config } from '../../../../src/core/config.ts';
import { Logger } from '../../../../src/core/logger.ts';
import { EventBus } from '../../../../src/core/event-bus.ts';

describe('Config Command Tests', () => {
  let configCommand: ConfigCommand;
  let mockConfig: Config;
  let mockLogger: Logger;
  let mockEventBus: EventBus;
  let consoleLogSpy: any;

  const sampleConfig = {
    apiKey: 'test-api-key',
    region: 'us-west-2',
    maxAgents: 10,
    timeout: 30000,
    debug: true,
    features: {
      autoSave: true,
      telemetry: false,
    },
  };

  beforeEach(() => {
    mockLogger = {
      info: spy(),
      error: spy(),
      warn: spy(),
      debug: spy(),
    } as unknown as Logger;

    mockConfig = {
      get: stub().callsFake((key: string) => {
        const keys = key.split('.');
        let value: any = sampleConfig;
        for (const k of keys) {
          value = value?.[k];
        }
        return value;
      }),
      set: stub().resolves(true),
      delete: stub().resolves(true),
      has: stub().callsFake((key: string) => {
        const value = mockConfig.get(key);
        return value !== undefined;
      }),
      getAll: stub().returns(sampleConfig),
      reset: stub().resolves(true),
      validate: stub().returns({ valid: true, errors: [] }),
      export: stub().returns(JSON.stringify(sampleConfig, null, 2)),
      import: stub().resolves(true),
      backup: stub().resolves('/path/to/backup.json'),
      restore: stub().resolves(true),
    } as unknown as Config;

    mockEventBus = {
      emit: spy(),
      on: spy(),
      off: spy(),
    } as unknown as EventBus;

    configCommand = new ConfigCommand(mockConfig, mockLogger, mockEventBus);
    consoleLogSpy = spy(console, 'log');
  });

  afterEach(() => {
    consoleLogSpy.restore();
  });

  describe('show subcommand', () => {
    it('should show all configuration', async () => {
      await configCommand.execute(['show']);
      
      assertSpyCall(mockConfig.getAll, 0);
      assertSpyCall(consoleLogSpy, 0);
      const output = consoleLogSpy.calls[0].args[0];
      assertEquals(typeof output, 'string');
      assertEquals(output.includes('apiKey'), true);
    });

    it('should mask sensitive values', async () => {
      await configCommand.execute(['show']);
      
      const output = consoleLogSpy.calls[0].args[0];
      assertEquals(output.includes('test-api-key'), false);
      assertEquals(output.includes('****'), true);
    });

    it('should show config in JSON format', async () => {
      await configCommand.execute(['show', '--format', 'json']);
      
      const output = JSON.parse(consoleLogSpy.calls[0].args[0]);
      assertEquals(output.region, 'us-west-2');
      assertEquals(output.apiKey, '****'); // Should be masked
    });

    it('should show config in YAML format', async () => {
      await configCommand.execute(['show', '--format', 'yaml']);
      
      const output = consoleLogSpy.calls[0].args[0];
      assertEquals(output.includes('region: us-west-2'), true);
      assertEquals(output.includes('apiKey: ****'), true);
    });
  });

  describe('get subcommand', () => {
    it('should get a specific config value', async () => {
      await configCommand.execute(['get', 'region']);
      
      assertSpyCall(mockConfig.get, 0, { args: ['region'] });
      assertSpyCall(consoleLogSpy, 0, { args: ['us-west-2'] });
    });

    it('should get nested config value', async () => {
      await configCommand.execute(['get', 'features.autoSave']);
      
      assertSpyCall(mockConfig.get, 0, { args: ['features.autoSave'] });
      assertSpyCall(consoleLogSpy, 0, { args: ['true'] });
    });

    it('should handle non-existent key', async () => {
      mockConfig.get = stub().returns(undefined);
      
      await configCommand.execute(['get', 'nonexistent']);
      
      assertSpyCall(mockLogger.warn, 0, {
        args: ['Config key "nonexistent" not found'],
      });
    });

    it('should support default value for missing keys', async () => {
      mockConfig.get = stub().returns(undefined);
      
      await configCommand.execute(['get', 'nonexistent', '--default', 'fallback']);
      
      assertSpyCall(consoleLogSpy, 0, { args: ['fallback'] });
    });
  });

  describe('set subcommand', () => {
    it('should set a config value', async () => {
      await configCommand.execute(['set', 'region', 'eu-west-1']);
      
      assertSpyCall(mockConfig.set, 0, {
        args: ['region', 'eu-west-1'],
      });
      assertSpyCall(mockLogger.info, 0, {
        args: ['Config updated: region = eu-west-1'],
      });
    });

    it('should set nested config value', async () => {
      await configCommand.execute(['set', 'features.autoSave', 'false']);
      
      assertSpyCall(mockConfig.set, 0, {
        args: ['features.autoSave', false], // Should parse boolean
      });
    });

    it('should parse numeric values', async () => {
      await configCommand.execute(['set', 'maxAgents', '20']);
      
      assertSpyCall(mockConfig.set, 0, {
        args: ['maxAgents', 20], // Should parse number
      });
    });

    it('should parse boolean values', async () => {
      await configCommand.execute(['set', 'debug', 'false']);
      
      assertSpyCall(mockConfig.set, 0, {
        args: ['debug', false],
      });
    });

    it('should validate config after setting', async () => {
      await configCommand.execute(['set', 'region', 'invalid-region']);
      
      assertSpyCall(mockConfig.validate, 0);
    });

    it('should handle validation errors', async () => {
      mockConfig.validate = stub().returns({
        valid: false,
        errors: ['Invalid region: invalid-region'],
      });
      
      await assertRejects(
        async () => await configCommand.execute(['set', 'region', 'invalid-region']),
        Error,
        'Config validation failed'
      );
    });

    it('should emit config.changed event', async () => {
      await configCommand.execute(['set', 'region', 'eu-west-1']);
      
      assertSpyCall(mockEventBus.emit, 0, {
        args: ['config.changed', { key: 'region', oldValue: 'us-west-2', newValue: 'eu-west-1' }],
      });
    });
  });

  describe('delete subcommand', () => {
    it('should delete a config key', async () => {
      await configCommand.execute(['delete', 'debug']);
      
      assertSpyCall(mockConfig.delete, 0, { args: ['debug'] });
      assertSpyCall(mockLogger.info, 0, {
        args: ['Config key "debug" deleted'],
      });
    });

    it('should require confirmation for sensitive keys', async () => {
      const readLineSpy = stub(Deno.stdin, 'readSync').returns(new Uint8Array([121, 10])); // 'y\n'
      
      await configCommand.execute(['delete', 'apiKey']);
      
      assertSpyCall(mockLogger.warn, 0, {
        args: ['Are you sure you want to delete "apiKey"? This action cannot be undone. (y/N)'],
      });
      assertSpyCall(mockConfig.delete, 0, { args: ['apiKey'] });
      
      readLineSpy.restore();
    });

    it('should not delete if confirmation is denied', async () => {
      const readLineSpy = stub(Deno.stdin, 'readSync').returns(new Uint8Array([110, 10])); // 'n\n'
      
      await configCommand.execute(['delete', 'apiKey']);
      
      assertSpyCalls(mockConfig.delete, 0); // Should not be called
      assertSpyCall(mockLogger.info, 0, {
        args: ['Deletion cancelled'],
      });
      
      readLineSpy.restore();
    });
  });

  describe('reset subcommand', () => {
    it('should reset all configuration with confirmation', async () => {
      const readLineSpy = stub(Deno.stdin, 'readSync').returns(new Uint8Array([121, 10])); // 'y\n'
      
      await configCommand.execute(['reset']);
      
      assertSpyCall(mockLogger.warn, 0, {
        args: ['Are you sure you want to reset all configuration? This action cannot be undone. (y/N)'],
      });
      assertSpyCall(mockConfig.reset, 0);
      assertSpyCall(mockLogger.info, 0, {
        args: ['Configuration reset to defaults'],
      });
      
      readLineSpy.restore();
    });

    it('should skip confirmation with --force flag', async () => {
      await configCommand.execute(['reset', '--force']);
      
      assertSpyCall(mockConfig.reset, 0);
      assertSpyCalls(mockLogger.warn, 0); // No warning shown
    });
  });

  describe('validate subcommand', () => {
    it('should validate configuration', async () => {
      await configCommand.execute(['validate']);
      
      assertSpyCall(mockConfig.validate, 0);
      assertSpyCall(mockLogger.info, 0, {
        args: ['Configuration is valid'],
      });
    });

    it('should show validation errors', async () => {
      mockConfig.validate = stub().returns({
        valid: false,
        errors: ['Missing required field: apiKey', 'Invalid region format'],
      });
      
      await configCommand.execute(['validate']);
      
      assertSpyCall(mockLogger.error, 0, {
        args: ['Configuration validation failed:'],
      });
      assertSpyCalls(mockLogger.error, 3); // Header + 2 errors
    });
  });

  describe('export/import subcommands', () => {
    it('should export configuration to file', async () => {
      const writeTextFileSpy = stub(Deno, 'writeTextFile').resolves();
      
      await configCommand.execute(['export', '/tmp/config.json']);
      
      assertSpyCall(mockConfig.export, 0);
      assertSpyCall(writeTextFileSpy, 0, {
        args: ['/tmp/config.json', JSON.stringify(sampleConfig, null, 2)],
      });
      assertSpyCall(mockLogger.info, 0, {
        args: ['Configuration exported to /tmp/config.json'],
      });
      
      writeTextFileSpy.restore();
    });

    it('should export to stdout if no file specified', async () => {
      await configCommand.execute(['export']);
      
      assertSpyCall(consoleLogSpy, 0);
      const output = consoleLogSpy.calls[0].args[0];
      assertEquals(typeof output, 'string');
    });

    it('should import configuration from file', async () => {
      const readTextFileSpy = stub(Deno, 'readTextFile').resolves(JSON.stringify(sampleConfig));
      
      await configCommand.execute(['import', '/tmp/config.json']);
      
      assertSpyCall(readTextFileSpy, 0, { args: ['/tmp/config.json'] });
      assertSpyCall(mockConfig.import, 0, { args: [sampleConfig] });
      assertSpyCall(mockLogger.info, 0, {
        args: ['Configuration imported from /tmp/config.json'],
      });
      
      readTextFileSpy.restore();
    });

    it('should validate imported configuration', async () => {
      const readTextFileSpy = stub(Deno, 'readTextFile').resolves(JSON.stringify(sampleConfig));
      mockConfig.validate = stub().returns({ valid: false, errors: ['Invalid config'] });
      
      await assertRejects(
        async () => await configCommand.execute(['import', '/tmp/config.json']),
        Error,
        'Imported configuration is invalid'
      );
      
      readTextFileSpy.restore();
    });
  });

  describe('backup/restore subcommands', () => {
    it('should create configuration backup', async () => {
      await configCommand.execute(['backup']);
      
      assertSpyCall(mockConfig.backup, 0);
      assertSpyCall(mockLogger.info, 0, {
        args: ['Configuration backed up to /path/to/backup.json'],
      });
    });

    it('should create backup with custom name', async () => {
      await configCommand.execute(['backup', '--name', 'before-update']);
      
      assertSpyCall(mockConfig.backup, 0, { args: ['before-update'] });
    });

    it('should restore configuration from backup', async () => {
      await configCommand.execute(['restore', 'backup-123']);
      
      assertSpyCall(mockConfig.restore, 0, { args: ['backup-123'] });
      assertSpyCall(mockLogger.info, 0, {
        args: ['Configuration restored from backup: backup-123'],
      });
    });

    it('should list available backups', async () => {
      mockConfig.listBackups = stub().returns([
        { name: 'backup-1', date: new Date('2024-01-01'), size: 1024 },
        { name: 'backup-2', date: new Date('2024-01-02'), size: 2048 },
      ]);
      
      await configCommand.execute(['backup', '--list']);
      
      assertSpyCalls(consoleLogSpy, 3); // Header + 2 backups
    });
  });

  describe('error handling', () => {
    it('should handle file write errors on export', async () => {
      const writeTextFileSpy = stub(Deno, 'writeTextFile').rejects(new Error('Permission denied'));
      
      await assertRejects(
        async () => await configCommand.execute(['export', '/root/config.json']),
        Error,
        'Permission denied'
      );
      
      writeTextFileSpy.restore();
    });

    it('should handle file read errors on import', async () => {
      const readTextFileSpy = stub(Deno, 'readTextFile').rejects(new Error('File not found'));
      
      await assertRejects(
        async () => await configCommand.execute(['import', '/tmp/nonexistent.json']),
        Error,
        'File not found'
      );
      
      readTextFileSpy.restore();
    });

    it('should handle invalid JSON on import', async () => {
      const readTextFileSpy = stub(Deno, 'readTextFile').resolves('invalid json');
      
      await assertRejects(
        async () => await configCommand.execute(['import', '/tmp/invalid.json']),
        Error,
        'Invalid JSON in configuration file'
      );
      
      readTextFileSpy.restore();
    });
  });

  describe('advanced features', () => {
    it('should support environment variable interpolation', async () => {
      Deno.env.set('TEST_API_KEY', 'env-api-key');
      
      await configCommand.execute(['set', 'apiKey', '${TEST_API_KEY}']);
      
      assertSpyCall(mockConfig.set, 0, {
        args: ['apiKey', 'env-api-key'],
      });
      
      Deno.env.delete('TEST_API_KEY');
    });

    it('should support config profiles', async () => {
      await configCommand.execute(['profile', 'development']);
      
      assertSpyCall(mockConfig.setProfile, 0, { args: ['development'] });
      assertSpyCall(mockLogger.info, 0, {
        args: ['Switched to profile: development'],
      });
    });

    it('should encrypt sensitive values', async () => {
      await configCommand.execute(['set', 'apiKey', 'secret-key', '--encrypt']);
      
      assertSpyCall(mockConfig.set, 0);
      const setValue = mockConfig.set.calls[0].args[1];
      assertEquals(setValue.startsWith('enc:'), true); // Should be encrypted
    });
  });
});