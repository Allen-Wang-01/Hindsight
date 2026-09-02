/**
 * Tool definitions handed to the Responses API.
 *
 * These descriptions are prompt engineering, not documentation — they are the
 * only thing the model reads when deciding which tool to reach for, so they
 * carry the constraints that shape its behaviour: summaries versus full text,
 * cost ordering, and the fact that a saved record is not the user's own work.
 *
 * `strict` is deliberately false. Strict schemas would let the API reject
 * malformed arguments for us, which sounds good until phase 5, where tool
 * argument error rate is a reported metric — enforcement upstream would pin
 * that number at zero and hide the behaviour we are trying to measure. The
 * stubs validate their own arguments instead.
 */
import type OpenAI from 'openai'

export const AGENT_TOOLS: OpenAI.Responses.Tool[] = [
    {
        type: 'function',
        name: 'search_records',
        strict: false,
        description:
            'Semantic search over the user\'s own records. Returns SUMMARIES ONLY — ' +
            'if a summary looks like it matters, call get_record to read the full text. ' +
            'Each result carries a `relatedness` of "strong" or "weak"; use it to decide ' +
            'what is worth opening, and do not report it as a score in your narration. ' +
            'Returns count 0 when nothing matches, which is a valid answer, not a failure. ' +
            'Reach for this when you need to find records you cannot name in advance.',
        parameters: {
            type: 'object',
            properties: {
                query: {
                    type: 'string',
                    description: 'Natural-language description of what you are looking for.',
                },
                date_range: {
                    type: 'object',
                    description: 'Optional ISO 8601 bounds on when the record happened.',
                    properties: {
                        from: { type: 'string' },
                        to: { type: 'string' },
                    },
                },
            },
            required: ['query'],
        },
    },
    {
        type: 'function',
        name: 'get_unit_history',
        strict: false,
        description:
            'Past evidence recorded against one cognitive unit, oldest first. ' +
            'A "dimension" is a broad capability; a "subject" is a specific topic. ' +
            '`unit` accepts either the unit id or its name, so you can call this with ' +
            'the name you already have. Returns EXCERPTS ONLY — call get_record for the ' +
            'full text behind any entry. ' +
            'A unit with no history returns count 0 and a null unit_id: that means this ' +
            'is the first time the user has been observed here, NOT that anything failed, ' +
            'and NOT that they have regressed. Reach for this when you want to know ' +
            'whether what you are looking at has happened before.',
        parameters: {
            type: 'object',
            properties: {
                unit_type: {
                    type: 'string',
                    enum: ['dimension', 'subject'],
                    description: 'Whether `unit` names a broad capability or a specific topic.',
                },
                unit: {
                    type: 'string',
                    description: 'The unit id, or its name.',
                },
            },
            required: ['unit_type', 'unit'],
        },
    },
    {
        type: 'function',
        name: 'get_topic_distribution',
        strict: false,
        description:
            'How often each subject has come up in a recent window, with the date it was ' +
            'last seen. This is the cheapest tool available — prefer it when the question ' +
            'is "what has the user been working on lately" rather than "find me a specific ' +
            'record". It returns counts and dates only, never record text.',
        parameters: {
            type: 'object',
            properties: {
                dimension: {
                    type: 'string',
                    description:
                        'Optional. Restrict to subjects under one dimension. ' +
                        'Omit to see every dimension at once.',
                },
                window: {
                    type: 'string',
                    description: 'Optional lookback window such as "30d" or "90d". Defaults to "90d".',
                },
            },
            required: [],
        },
    },
    {
        type: 'function',
        name: 'get_record',
        strict: false,
        description:
            'The full text of one record. This is the ONLY tool that returns full text, ' +
            'and it takes an id you must have obtained from another tool first. ' +
            'Read `source_type` before you use anything you find here: "authored" means the ' +
            'user wrote it themselves, so it is evidence of what they did. "saved" means they ' +
            'preserved someone else\'s material — it tells you what they were trying to learn, ' +
            'and NEVER what they can do. Attributing saved content to the user is a serious ' +
            'error. When `description` is present it is the user\'s own reason for keeping the ' +
            'record, and it outranks anything you infer from the content.',
        parameters: {
            type: 'object',
            properties: {
                record_id: {
                    type: 'string',
                    description: 'Record id returned by search_records or get_unit_history.',
                },
            },
            required: ['record_id'],
        },
    },
]
