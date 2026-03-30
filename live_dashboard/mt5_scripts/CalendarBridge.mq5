//+------------------------------------------------------------------+
//| CalendarBridge.mq5 — Exports upcoming economic events to JSON    |
//| Install: Compile in MetaEditor, attach as Service to any chart   |
//| Output: Common/Files/calendar_events.json (every 15 minutes)     |
//+------------------------------------------------------------------+
#property service
#property copyright "Spartus Trading AI"
#property version   "1.00"
#property description "Exports MT5 economic calendar events to JSON for the live dashboard"

// Output file path (in MQL5 Common Files folder)
input string OutputFile = "calendar_events.json";
// Update interval in seconds (900 = 15 minutes)
input int UpdateIntervalSeconds = 900;
// How many days ahead to look for events
input int LookAheadDays = 7;

//+------------------------------------------------------------------+
//| Service entry point                                               |
//+------------------------------------------------------------------+
void OnStart()
{
    Print("CalendarBridge started. Output: ", OutputFile,
          ", Interval: ", UpdateIntervalSeconds, "s",
          ", Lookahead: ", LookAheadDays, " days");

    while(!IsStopped())
    {
        ExportCalendarEvents();
        Sleep(UpdateIntervalSeconds * 1000);
    }

    Print("CalendarBridge stopped.");
}

//+------------------------------------------------------------------+
//| Export upcoming calendar events to JSON                           |
//+------------------------------------------------------------------+
void ExportCalendarEvents()
{
    // Time range: now to N days ahead
    datetime from_time = TimeCurrent();
    datetime to_time = from_time + LookAheadDays * 86400;

    // Get calendar values
    MqlCalendarValue values[];
    int count = CalendarValueHistory(values, from_time, to_time);

    if(count < 0)
    {
        Print("CalendarValueHistory failed, error: ", GetLastError());
        return;
    }

    // Open file for writing
    int handle = FileOpen(OutputFile, FILE_WRITE | FILE_TXT | FILE_COMMON | FILE_ANSI);
    if(handle == INVALID_HANDLE)
    {
        Print("Failed to open file: ", OutputFile, ", error: ", GetLastError());
        return;
    }

    // Write JSON header
    string now_str = TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS);
    StringReplace(now_str, ".", "-");
    FileWriteString(handle, "{\n");
    FileWriteString(handle, "  \"updated_at\": \"" + now_str + "\",\n");
    FileWriteString(handle, "  \"events\": [\n");

    int written = 0;

    for(int i = 0; i < count; i++)
    {
        // Get event details
        MqlCalendarEvent event;
        if(!CalendarEventById(values[i].event_id, event))
            continue;

        // Filter: only HIGH importance events
        if(event.importance != CALENDAR_IMPORTANCE_HIGH)
            continue;

        // Get country info for currency
        MqlCalendarCountry country;
        if(!CalendarCountryById(event.country_id, country))
            continue;

        // Filter currencies we care about: USD, EUR, GBP, JPY, AUD, CAD, CHF
        string currency = country.currency;
        if(currency != "USD" && currency != "EUR" && currency != "GBP" &&
           currency != "JPY" && currency != "AUD" && currency != "CAD" &&
           currency != "CHF")
            continue;

        // Format event time
        string event_time = TimeToString(values[i].time, TIME_DATE | TIME_SECONDS);
        StringReplace(event_time, ".", "-");

        // Format values
        string forecast_str = (values[i].forecast_value != LONG_MAX) ?
            DoubleToString(values[i].forecast_value / 1000000.0, 3) : "null";
        string previous_str = (values[i].prev_value != LONG_MAX) ?
            DoubleToString(values[i].prev_value / 1000000.0, 3) : "null";
        string actual_str = (values[i].actual_value != LONG_MAX) ?
            DoubleToString(values[i].actual_value / 1000000.0, 3) : "null";

        // Importance string
        string importance = "HIGH";

        // Write comma separator (not before first entry)
        if(written > 0)
            FileWriteString(handle, ",\n");

        // Write event JSON
        FileWriteString(handle, "    {");
        FileWriteString(handle, "\"time\": \"" + event_time + "\", ");
        FileWriteString(handle, "\"name\": \"" + event.name + "\", ");
        FileWriteString(handle, "\"currency\": \"" + currency + "\", ");
        FileWriteString(handle, "\"importance\": \"" + importance + "\", ");
        FileWriteString(handle, "\"forecast\": " + forecast_str + ", ");
        FileWriteString(handle, "\"previous\": " + previous_str + ", ");
        FileWriteString(handle, "\"actual\": " + actual_str);
        FileWriteString(handle, "}");

        written++;
    }

    // Close JSON
    FileWriteString(handle, "\n  ]\n");
    FileWriteString(handle, "}\n");
    FileClose(handle);

    Print("CalendarBridge: exported ", written, " high-impact events");
}
//+------------------------------------------------------------------+
