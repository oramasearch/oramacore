# Operation transport

RabbitMQ consumers reconnect from the read side's next unapplied offset. Closing
the old consumer is best effort and bounded. A separate short-lived consumer at
`OffsetSpecification::Last` reads the last chunk to determine stream head, with
bounded creation, idle reads and close. Its offset is observation only; it never
advances the reader's apply position. In-memory receivers have no remote head.
