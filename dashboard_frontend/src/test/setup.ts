import '@testing-library/jest-dom';

/**
 * Node 22+ can expose a broken experimental `localStorage` when
 * `NODE_OPTIONS` includes `--localstorage-file` without a valid path
 * (warning: "`--localstorage-file` was provided without a valid path").
 * That stub shadows jsdom's storage and breaks tests that call
 * `clear()` / `setItem()`. Use a simple in-memory Storage instead.
 */
function createMemoryStorage(): Storage {
  const store = new Map<string, string>();

  return {
    get length() {
      return store.size;
    },
    clear() {
      store.clear();
    },
    getItem(key: string) {
      return store.has(key) ? store.get(key)! : null;
    },
    key(index: number) {
      return [...store.keys()][index] ?? null;
    },
    removeItem(key: string) {
      store.delete(key);
    },
    setItem(key: string, value: string) {
      store.set(key, String(value));
    },
  };
}

if (typeof window !== 'undefined') {
  Object.defineProperty(window, 'localStorage', {
    value: createMemoryStorage(),
    configurable: true,
    writable: true,
  });
}
