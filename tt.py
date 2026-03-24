from tools import browser_navigate_tool, register_conversation_urls

register_conversation_urls('https://www.linkedin.com/feed')

result = browser_navigate_tool(
     url='https://www.linkedin.com/feed',
     action='test_cdp_x',
     use_cdp=True,
     wait_until='domcontentloaded',
     wait_ms=3000,  # Extra wait for JS to render
 )
print(result)
