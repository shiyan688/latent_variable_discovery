# Agent handoff instructions

Before doing material work in this repository, read [AGENT_MEMORY.md](AGENT_MEMORY.md) in full. It is the explicit handoff record for the ongoing latent-variable research and GPU campaign.

When taking over:

1. Treat the live JSON files and raw `result.json` files as the source of truth; the status snapshot in the memory document is intentionally timestamped and may be stale.
2. Refresh `runs/extended_15h_campaign_20260809/campaign_status.json`, the relevant `tmux` sessions, and GPU availability before reporting or changing an experiment.
3. Preserve the dirty worktree. Do not reset, discard, or overwrite pre-existing modifications.
4. Do not terminate another user's jobs or claim a busy GPU. The campaign has an availability guard and is expected to wait when cards are occupied.
5. Update `AGENT_MEMORY.md` after a material experiment transition, result synthesis, protocol change, or newly discovered blocker. Separate timestamped observations from durable conclusions.
6. Do not promote partial extended-campaign results to confirmatory claims. Run the frozen analysis after the campaign reaches a terminal state and reconcile all claims with raw outputs.
